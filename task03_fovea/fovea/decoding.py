"""Optional DARK-inspired inference; training retains integral regression.

This is an adaptation for spatial-softmax logits, not an exact reproduction of
the DARK MSE-heatmap training pipeline. Pixel centers match geometry.py.
"""
import math
import torch
from torch.nn import functional as F


def decode_heatmap(logits, method='integral', blur_sigma=1.0):
    if logits.ndim!=4 or logits.shape[1]!=1:
        raise ValueError('Expected Bx1xHxW logits.')
    if method not in ('integral','dark'):
        raise ValueError(f'Unknown decoder: {method}')
    if not math.isfinite(blur_sigma) or blur_sigma<0:
        raise ValueError('blur_sigma must be finite and nonnegative.')
    values=logits[:,0].float()
    b,h,w=values.shape
    prob=values.flatten(1).softmax(-1).reshape(b,h,w)
    gx=(torch.arange(w,device=logits.device,dtype=torch.float32)+0.5)/w
    gy=(torch.arange(h,device=logits.device,dtype=torch.float32)+0.5)/h
    integral=torch.stack([(prob.sum(1)*gx).sum(1),(prob.sum(2)*gy).sum(1)],1)
    if method=='integral' or min(h,w)<5:
        return integral
    # Exponentiate logits before smoothing; smoothing logits would mix the
    # log density rather than the density used in the DARK derivation.
    density=(values-values.amax((-2,-1),keepdim=True)).exp()[:,None]
    if blur_sigma:
        radius=min(int(math.ceil(3*blur_sigma)),(min(h,w)-1)//2)
        axis=torch.arange(-radius,radius+1,device=logits.device,dtype=torch.float32)
        kernel=(-axis.square()/(2*blur_sigma**2)).exp()
        kernel=kernel/kernel.sum()
        density=F.conv2d(F.pad(density,(radius,radius,0,0),mode='replicate'),kernel[None,None,None,:])
        density=F.conv2d(F.pad(density,(0,0,radius,radius),mode='replicate'),kernel[None,None,:,None])
    logdensity=density[:,0].clamp_min(1e-12).log()
    peak=density.flatten(1).argmax(1)
    px,py=peak%w,peak//w
    ix,iy=px.clamp(1,w-2),py.clamp(1,h-2)
    row=torch.arange(b,device=logits.device)
    def at(dx,dy):
        return logdensity[row,iy+dy,ix+dx]
    dx=(at(1,0)-at(-1,0))*0.5
    dy=(at(0,1)-at(0,-1))*0.5
    dxx=at(1,0)-2*at(0,0)+at(-1,0)
    dyy=at(0,1)-2*at(0,0)+at(0,-1)
    dxy=(at(1,1)-at(-1,1)-at(1,-1)+at(-1,-1))*0.25
    determinant=dxx*dyy-dxy.square()
    denom=determinant.clamp_min(1e-12)
    sx=(dyy*dx-dxy*dy)/denom
    sy=(dxx*dy-dxy*dx)/denom
    stable=(determinant>1e-8)&(dxx<0)&(dyy<0)&(sx.abs()<=1.5)&(sy.abs()<=1.5)
    interior=(px>0)&(px<w-1)&(py>0)&(py<h-1)
    stable=stable&interior&torch.isfinite(sx)&torch.isfinite(sy)
    refined=torch.stack([px-torch.where(stable,sx,0),py-torch.where(stable,sy,0)],1)
    result=(refined+0.5)/refined.new_tensor([w,h])
    # Completely flat logits have no dominant mode; argmax would choose a
    # corner. Preserve the integral estimate for this degenerate case.
    flat=(values.amax((-2,-1))-values.amin((-2,-1)))<1e-6
    return torch.where(flat[:,None],integral,result)
