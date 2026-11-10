import torch
from torch.nn import functional as F


def dice_loss(prob, target):
    dims = tuple(range(1,prob.ndim))
    return (1 - (2*(prob*target).sum(dims)+1e-6)/(prob.sum(dims)+target.sum(dims)+1e-6)).mean()


def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)), -F.max_pool2d(-x,(1,3),1,(0,1)))


def soft_skeleton(x, iterations):
    opened = F.max_pool2d(soft_erode(x),3,1,1)
    skel = F.relu(x-opened)
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x-F.max_pool2d(soft_erode(x),3,1,1))
        skel = skel + F.relu(delta - skel*delta)
    return skel


def cldice_loss(prob, target, iterations):
    skel_p, skel_t = soft_skeleton(prob,iterations), soft_skeleton(target,iterations)
    dims = tuple(range(1,prob.ndim))
    precision = ((skel_p*target).sum(dims)+1e-6)/(skel_p.sum(dims)+1e-6)
    sensitivity = ((skel_t*prob).sum(dims)+1e-6)/(skel_t.sum(dims)+1e-6)
    return (1 - 2*precision*sensitivity/(precision+sensitivity+1e-6)).mean()


def segmentation_loss(outputs, target, config, epoch):
    # Force loss reductions into FP32, also when convolution uses AMP.
    logits = outputs['logits'].float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits,target)
    dice = dice_loss(logits.sigmoid(),target)
    ramp = min(1., max(0., (epoch-config['cldice_start_epoch'])/max(1,config['cldice_ramp_epochs'])))
    topology = cldice_loss(logits.sigmoid(),target,config['skeleton_iterations']) if ramp*config['cldice_weight'] > 0 else logits.new_zeros(())
    total = config['bce_weight']*bce + config['dice_weight']*dice + ramp*config['cldice_weight']*topology
    for weight, aux in zip(config['deep_supervision_weights'], outputs['aux']):
        resized = F.interpolate(aux.float(),size=target.shape[-2:],mode='bilinear',align_corners=False)
        total = total + weight*(0.5*F.binary_cross_entropy_with_logits(resized,target)+0.5*dice_loss(resized.sigmoid(),target))
    return total, {'bce':float(bce.detach()),'dice_loss':float(dice.detach()),'cldice_loss':float(topology.detach()),'cldice_ramp':ramp}
