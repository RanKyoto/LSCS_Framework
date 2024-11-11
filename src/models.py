from typing import Dict, List, Tuple, Type, Union, Callable
import torch as th
import torch.nn as nn
from src.networks import LyapunovFunction,create_mlp

class Stable_Dynamics(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 net_arch: List[int] = [32,32],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 device: Union[th.device, str] = "auto"
                 ):
        """
        Initializes a neural network with specified architecture, activation function,
        and device configuration.

        Parameters:
            state_dim (int): Dimension of the input state space, i.e., the number of input features.
            action_dim (int): Dimension of the action space, i.e., the number of output units.
            net_arch (List[int], optional): List specifying the number of units in each hidden layer
                of the neural network. Default is [32, 32], indicating two hidden layers with 32 units each.
            activation_fn (Type[nn.Module], optional): Activation function to be used between layers.
                Default is nn.ReLU. Should be a subclass of nn.Module.
            device (Union[th.device, str], optional): Specifies the device for computation (e.g., 'cpu',
                'cuda', or 'auto' to select automatically based on availability). Default is "auto".
        """
        super().__init__()
        """stable dynamics init"""
        self.device = device
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.x_stable = th.zeros([1,state_dim],dtype=th.float32, device=device).to(device)
        self.f_hat = create_mlp(state_dim,state_dim,net_arch).to(device)
        self.lyapunov_function = LyapunovFunction(state_dim,
                                                  net_arch,
                                                  activation_fn,
                                                  device=device)
        self.alpha_hat = create_mlp(state_dim,action_dim,net_arch).to(device)
        self.g = nn.ModuleList([create_mlp(state_dim,state_dim,net_arch).to(device) for _ in range(action_dim)])
    
    def forward(self, x:th.Tensor):
        f0 = (self.f_hat(x) - self.f_hat(self.x_stable)).unsqueeze(dim=2)
        alpha = (self.alpha_hat(x) - self.alpha_hat(self.x_stable)).unsqueeze(dim=2)
        g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        #g = th.tensor([0.0,1.0],dtype=th.float32,device=self.device).reshape((1,2,1))
        
        V = self.lyapunov_function.forward(x)
        V.sum().backward(retain_graph=True)
        grad_V = x.grad.unsqueeze(dim=2)
        
        W = 0.1 * x.pow(2).sum(1,keepdim=True).unsqueeze(dim=2)

        criterion = grad_V.transpose(1,2) @ (f0 + g @ alpha) + W
        fs = -  criterion / (grad_V.pow(2).sum(1,keepdim=True) + 1e-6) * grad_V
        
        mask = criterion <= 0
        f = th.where(mask, f0, f0 + fs)

        del x
        return f, g, alpha, V
    
class Safty_Dynamics(nn.Module):
    def __init__(self,
                state_dim: int, 
                action_dim: int,
                eta:Callable,
                c4:float, 
                net_arch: List[int] = [32, 32],
                activation_fn: Type[nn.Module] = nn.ReLU, 
                device: Union[th.device, str] = "auto"
                ):
         
        super().__init__()
        """stable dynamics init"""
        self.device = device
        self.state_dim = state_dim 
        self.action_dim = action_dim

        self.eta = eta
        self.c4 = c4
    
        self.f_hat = create_mlp(state_dim,state_dim,net_arch).to(device)
        self.alpha = create_mlp(state_dim,action_dim,net_arch).to(device)
        self.g = nn.ModuleList([create_mlp(state_dim,state_dim,net_arch).to(device) for _ in range(action_dim)])

    def forward(self, x:th.Tensor):
        f_hat = self.f_hat(x).unsqueeze(dim=2)
        alpha = self.alpha(x).unsqueeze(dim=2)
        g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        
        eta = self.eta(x).reshape(-1,1,1)
        eta.sum().backward(retain_graph=True)
        grad_eta = x.grad.unsqueeze(dim=2)

        criterion = grad_eta.transpose(1,2) @ (f_hat + g @ alpha) + self.c4 * eta
        f_eta = -  criterion / (grad_eta.pow(2).sum(1,keepdim=True) + 1e-6) * grad_eta
        
        mask = criterion <= 0
        f = th.where(mask, f_hat, f_hat + f_eta)

        del x
        return f, g, alpha, eta

class L2_Dynamics(Stable_Dynamics):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 net_arch: List[int] = [32,32],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 device: Union[th.device, str] = "auto"
                 ):
        super().__init__(state_dim, action_dim, net_arch, activation_fn, device)
         
    def forward(self, x:th.Tensor):
        f0 = (self.f_hat(x) - self.f_hat(self.x_stable)).unsqueeze(dim=2)
        alpha = (self.alpha_hat(x) - self.alpha_hat(self.x_stable)).unsqueeze(dim=2)
        #g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        g = th.tensor([0.0,1.0],dtype=th.float32,device=self.device).reshape((1,2,1))
        
        V = self.lyapunov_function.forward(x)
        grad_V = th.autograd.grad([V.sum()], [x], create_graph=True)[0]
        if grad_V is not None:
            grad_V = grad_V.reshape(-1,self.state_dim,1)
        else:
            grad_V = th.zeros_like(x,dtype=th.float32,device=self.device)
        
        # please modify W for different dynamics, this is only for van der Pol
        W = (x[:,0] ** 2).reshape(-1,1,1) + grad_V[:,1,:].pow(2).reshape(-1,1,1)

        criterion = grad_V.transpose(1,2) @ (f0 + g @ alpha) +  W
        fs = -  criterion / (grad_V.pow(2).sum(1,keepdim=True) + 1e-6) * grad_V
        
        mask = criterion <= 0
        f = th.where(mask, f0, f0 + fs)

        del x
        return f, g, alpha, V
    
class Passification_Dynamics(Stable_Dynamics):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 net_arch: List[int] = [32,32],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 device: Union[th.device, str] = "auto"
                 ):
        super().__init__(state_dim, action_dim, net_arch, activation_fn, device)
        self.beta = nn.ModuleList([create_mlp(state_dim,action_dim,net_arch).to(device) for _ in range(action_dim)])
        
    def forward(self, x:th.Tensor):
        f0 = (self.f_hat(x) - self.f_hat(self.x_stable)).unsqueeze(dim=2)
        alpha = (self.alpha_hat(x) - self.alpha_hat(self.x_stable)).unsqueeze(dim=2)
        g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        beta = th.stack([betai(x).reshape(-1, self.action_dim,1) for betai in self.beta],dim=2).squeeze(dim=3)
        
        V = self.lyapunov_function.forward(x)
        V.sum().backward(retain_graph=True)
        grad_V = x.grad.unsqueeze(dim=2)
        
        # please modify W for different dynamics, this is only for van der Pol
        W = (x[:,0] ** 2).reshape(-1,1,1) + grad_V[:,1,:].reshape(-1,1,1)

        criterion = grad_V.transpose(1,2) @ (f0 + g @ alpha) + W
        h = grad_V.transpose(1,2) @ (g @ beta)

        fs = -  criterion / (th.norm(grad_V,2,dim=1,keepdim=True) ** 2 + 1e-6) * grad_V
        
        mask = criterion <= 0
        f = th.where(mask, f0, f0 + fs)

        del x
        return f, g, alpha, V, h, beta

class HJI_Dynamics(Stable_Dynamics):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 R_fnc: Callable,
                 q_fnc: Callable,
                 net_arch: List[int] = [32,32],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 device: Union[th.device, str] = "auto"
                 ):
        super().__init__(state_dim, action_dim, net_arch, activation_fn, device)
        self.alpha_hat = None
        
        self.R = R_fnc
        self.q = q_fnc
        
    def forward_general(self, x:th.Tensor):
        f0 = (self.f_hat(x) - self.f_hat(self.x_stable)).unsqueeze(dim=2)
        #g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        g = th.ones_like(f0,dtype=th.float32,device=self.device)
        
        V = self.lyapunov_function.forward(x)
        V.sum().backward(retain_graph=True)
        grad_V = x.grad.unsqueeze(dim=2)
        
        LgV = grad_V.transpose(1,2) @ g
        alpha =  -th.linalg.inv(self.R(x).unsqueeze(dim=2)) @ LgV.transpose(1,2) 
        W = self.q(x).unsqueeze(dim=2) - 0.5 *  LgV @ alpha

        criterion = grad_V.transpose(1,2) @ (f0 + g @ alpha) + W
        fs = -  criterion / (th.norm(grad_V,2,dim=1,keepdim=True) ** 2 + 1e-3) * grad_V
        
        mask = criterion <= 0
        f = th.where(mask, f0, f0 + fs)

        H = grad_V.transpose(1,2) @ (f + g @ alpha) + W

        del x
        return f, g, alpha, V, H
    
    def forward(self, x:th.Tensor):
        f0 = (self.f_hat(x) - self.f_hat(self.x_stable)).unsqueeze(dim=2)
        #g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        g = th.ones_like(f0,dtype=th.float32,device=self.device)
   
        V = self.lyapunov_function.forward(x)
        
        grad_V = th.autograd.grad([V.sum()], [x], create_graph=True)[0]
        if grad_V is not None:
            grad_V = grad_V.reshape(-1,1,1)
        else:
            grad_V = th.zeros_like(x,dtype=th.float32,device=self.device)
        LgV = grad_V.transpose(1,2) @ g
        alpha =  - LgV.transpose(1,2) 
        #alpha = (-x*(x+th.sqrt(2+x**2))).unsqueeze(dim=2)
        W = (x ** 2).unsqueeze(dim=2) + 0.5 *  LgV ** 2

        criterion = grad_V.transpose(1,2) @ (f0 + g @ alpha) + W
        fs = -  criterion / (grad_V.pow(2).sum(1,keepdim=True) + 1e-3) * grad_V
        
        mask = criterion <= 0
        f = th.where(mask, f0, f0 + fs)

        H = grad_V.transpose(1,2) @ (f + g @ alpha) + W

       
        return f, g, alpha, V, H