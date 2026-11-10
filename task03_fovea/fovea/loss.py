import torch
from torch.nn import functional as F


def localization_loss(output, batch, sigma=1.7, coordinate_scale=100.0,
                      box_weight=0.15, presence_enabled=False, return_parts=True):
    logits = output['logits'][:,0]
    b,h,w = logits.shape
    y,x = torch.meshgrid(torch.arange(h,device=logits.device),torch.arange(w,device=logits.device),indexing='ij')
    target = batch['target']
    center = target*target.new_tensor([w,h])-0.5
    d2 = (x[None]-center[:,0,None,None])**2 + (y[None]-center[:,1,None,None])**2
    gaussian = torch.softmax((-d2/(2*sigma**2)).flatten(1),dim=1)
    logprob = F.log_softmax(logits.flatten(1),dim=1)
    kl = (gaussian*(gaussian.clamp_min(1e-12).log()-logprob)).sum(1)
    coordinate = (((output['xy']-target)*batch['side'][:,None]/coordinate_scale)**2).mean(1)
    weight = batch.get('sample_weight',torch.ones_like(batch['coord_valid']))
    valid = batch['coord_valid']*weight
    heat_loss = (kl*valid).sum()/valid.sum().clamp_min(1)
    xy_loss = (coordinate*valid).sum()/valid.sum().clamp_min(1)
    box_target = batch['box']
    box_bce = F.binary_cross_entropy_with_logits(output['box'],box_target,reduction='none',
                                               pos_weight=box_target.new_tensor(15)).mean((1,2,3))
    sigmoid = output['box'].sigmoid()
    dice = 1-(2*(sigmoid*box_target).sum((1,2,3))+1)/(sigmoid.sum((1,2,3))+box_target.sum((1,2,3))+1)
    bv = batch['box_valid']*weight
    box_loss = ((box_bce+dice)*bv).sum()/bv.sum().clamp_min(1)
    presence_loss = F.binary_cross_entropy_with_logits(output['presence'],batch['presence']) if presence_enabled else logits.sum()*0
    total = heat_loss+xy_loss+box_weight*box_loss+0.2*presence_loss
    if not return_parts:
        return total, {}
    return total, dict(heat=float(heat_loss.detach()),coordinate=float(xy_loss.detach()),
                       box=float(box_loss.detach()),presence=float(presence_loss.detach()))
