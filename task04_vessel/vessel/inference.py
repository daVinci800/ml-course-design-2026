import numpy as np
import torch


def transform(x, k, flip):
    x = torch.rot90(x,k,(-2,-1))
    return x.flip(-1) if flip else x


def inverse_transform(x, k, flip):
    x = x.flip(-1) if flip else x
    return torch.rot90(x,-k,(-2,-1))


@torch.inference_mode()
def predict_probability(model, tensor, device, tta=False, amp=True):
    model.eval()
    x = tensor[None].to(device)
    total = None
    variants = [(k,f) for k in range(4) for f in [False,True]] if tta else [(0,False)]
    for k, flip in variants:
        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
            prob = model(transform(x,k,flip))['logits'].float().sigmoid()
        prob = inverse_transform(prob,k,flip)
        total = prob if total is None else total+prob
    return (total[0,0]/len(variants)).cpu().numpy().astype(np.float32)


def binary_dice(prediction, target):
    p, t = np.asarray(prediction,dtype=bool), np.asarray(target,dtype=bool)
    return float((2*np.logical_and(p,t).sum()+1e-8)/(p.sum()+t.sum()+1e-8))
