import numpy as np
import matplotlib
import torch
import cufsm

class FSMCpuFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, f, iters):
        # Ensure float32 for C++ call
        src_f32 = src.to(torch.float32).contiguous()
        f_f32 = f.to(torch.float32).contiguous()
        
        N = src.shape[0]
        L = f.shape[0]
        
        T = torch.zeros((N, L, L), dtype=torch.float32, device=f.device).contiguous()
        cufsm.fsm_cpu(T, src_f32, f_f32, iters)
        
        ctx.save_for_backward(T, src_f32, f_f32)
        ctx.iters = iters
        ctx.dtype = f.dtype
        
        return T.to(f.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        T, src, f = ctx.saved_tensors
        iters = ctx.iters
        
        grad_output_f32 = grad_output.to(torch.float32).contiguous()
        grad_f = torch.zeros_like(f, dtype=torch.float32).contiguous()
        grad_src = torch.zeros_like(src, dtype=torch.float32).contiguous()
        
        cufsm.fsm_adjoint_cpu(T, grad_output_f32, src, f, grad_f, grad_src, iters)
        
        return grad_src.to(ctx.dtype), grad_f.to(ctx.dtype), None

class FSMGpuFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, f, iters):
        # assert src and f are on GPU
        assert src.is_cuda
        assert f.is_cuda

        # Ensure float32 for C++ call
        src_f32 = src.to(torch.float32).contiguous()
        f_f32 = f.to(torch.float32).contiguous()
        
        N = src.shape[0]
        L = f.shape[0]
        
        T = torch.zeros((N, L, L), dtype=torch.float32, device=f.device).contiguous()
        cufsm.fsm_gpu(T, src_f32, f_f32, iters)
        
        ctx.save_for_backward(T, src_f32, f_f32)
        ctx.iters = iters
        ctx.dtype = f.dtype
        
        return T.to(f.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        T, src, f = ctx.saved_tensors
        iters = ctx.iters
        
        grad_output_f32 = grad_output.to(torch.float32).contiguous()
        grad_f = torch.zeros_like(f, dtype=torch.float32).contiguous()
        grad_src = torch.zeros_like(src, dtype=torch.float32).contiguous()
        
        cufsm.fsm_adjoint_gpu(T, grad_output_f32, src, f, grad_f, grad_src, iters)
        
        return grad_src.to(ctx.dtype), grad_f.to(ctx.dtype), None

def soft_voronoi(T, beta=10):
    exp = torch.exp(-beta * T)
    return exp / exp.sum(dim=0, keepdim=True)

def rasterize_T_index(soft_T, index):
    L = soft_T.shape[1]
    return soft_T[index, :, :].reshape(-1, L, L)

def rasterize_T(soft_T, foregrounds):
    return torch.sum(soft_T[foregrounds, :, :], dim=0)

def render_soft_voronoi(soft_T, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ba21ce']):
    # convert soft_T to numpy
    soft_T = soft_T.detach().cpu().numpy()
    img = np.zeros((soft_T.shape[1], soft_T.shape[2], 3))
    for i in range(soft_T.shape[0]):
        color = matplotlib.colors.to_rgb(colors[i % len(colors)])
        for c in range(3):
            img[:, :, c] += soft_T[i] * color[c]
    img = np.clip(img, 0, 1)
    return img

def hard_voronoi(T):
    # convert T to numpy
    T_np = T.detach().cpu().numpy()
    ownerships = np.argmin(T_np, axis=0)
    return ownerships
