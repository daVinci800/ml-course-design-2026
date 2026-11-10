"""Confidence-weighted centers of overlapping hypotheses for the same object."""
import numpy as np


def dense_center(boxes,scores,iou_threshold=.5,relative_confidence=.2):
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4)
    scores=np.asarray(scores,dtype=float).reshape(-1)
    if len(boxes)!=len(scores): raise ValueError('Box-score count mismatch')
    valid=np.isfinite(boxes).all(1)&np.isfinite(scores)&(scores>0)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    if not valid.any(): return None
    boxes,scores=boxes[valid],scores[valid]
    best=boxes[np.argmax(scores)]
    intersect=np.maximum(0,np.minimum(boxes[:,2:],best[2:])-np.maximum(boxes[:,:2],best[:2])).prod(1)
    area=(boxes[:,2:]-boxes[:,:2]).prod(1);best_area=np.prod(best[2:]-best[:2])
    iou=intersect/(area+best_area-intersect)
    keep=(iou>=iou_threshold)&(scores>=relative_confidence*scores.max())
    return np.average((boxes[keep,:2]+boxes[keep,2:])/2,axis=0,weights=scores[keep]**2)
