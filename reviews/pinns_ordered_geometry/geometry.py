"""Six body-relative 2D image-geometry descriptors from observed keypoints."""
import itertools

import numpy as np

NAMES=('ankle_distance','left_knee_cosine','right_knee_cosine','shoulder_width','torso_offset_x','torso_height_y')
PERMUTATIONS=tuple(itertools.permutations(range(4)))


def descriptors(kpts,boxes):
    raw=np.asarray(kpts,dtype=np.float64);box=np.asarray(boxes,dtype=np.float64)
    if raw.ndim!=4 or raw.shape[1:]!=(5,17,3) or box.shape!=raw.shape[:2]+(4,):
        raise ValueError('Expected five observed COCO-17 poses and corresponding measured boxes')
    height=box[...,3]
    frame=np.isfinite(height)&(height>1.)
    good=(np.isfinite(raw).all(axis=-1)&(raw[...,2]>=.20)&frame[...,None])
    xy=np.where(good[...,None],raw[...,:2],0.)/np.where(frame,height,1.)[...,None,None]
    values=[];masks=[]
    def take(value,valid):
        masks.append(valid);values.append(np.where(valid,value,0.))
    def span(a,b):
        take(np.linalg.norm(xy[:,:,a]-xy[:,:,b],axis=-1),good[:,:,a]&good[:,:,b])
    span(15,16)
    for hip,knee,ankle in [(11,13,15),(12,14,16)]:
        thigh=xy[:,:,hip]-xy[:,:,knee];shin=xy[:,:,ankle]-xy[:,:,knee]
        a,b=np.linalg.norm(thigh,axis=-1),np.linalg.norm(shin,axis=-1)
        ok=good[:,:,hip]&good[:,:,knee]&good[:,:,ankle]&(a>1e-6)&(b>1e-6)
        cosine=(thigh*shin).sum(-1)/np.where(ok,a*b,1.)
        take(np.clip(cosine,-1.,1.),ok)
    span(5,6)
    trunk=(xy[:,:,5]+xy[:,:,6]-xy[:,:,11]-xy[:,:,12])*.5
    combined=good[:,:,[5,6,11,12]].all(-1)
    take(trunk[...,0],combined);take(-trunk[...,1],combined)
    return np.stack(values,-1),np.stack(masks,-1)


def design_features(motion,geometry,valid):
    if geometry.shape!=valid.shape or geometry.shape[1:]!=(5,6) or motion.shape!=(len(valid),28):
        raise ValueError('Invalid history features')
    common=valid.all(axis=1)
    base=np.column_stack((motion,geometry[:,-1],valid[:,-1],common))
    increments=np.where(common[:,None],np.diff(geometry,axis=1),0.)
    static=np.column_stack((base,np.zeros((len(base),24))))
    ordered=np.column_stack((base,increments.reshape(len(base),24)))
    return static,ordered,increments


def permuted(ordered,permutation):
    if ordered.ndim!=2 or ordered.shape[1]!=70 or tuple(sorted(permutation))!=(0,1,2,3):
        raise ValueError('Expected four complete vector slots')
    x=ordered.copy()
    x[:,-24:]=ordered[:,-24:].reshape(-1,4,6)[:,permutation].reshape(-1,24)
    return x


def blind_input(ordered):
    """Reynolds average of 24 input permutations; equivalent for a frozen linear predictor."""
    if ordered.ndim!=2 or ordered.shape[1]!=70:raise ValueError('Expected 70 feature columns')
    x=ordered.copy()
    # Summing a canonical ordering of the four numbers gives bitwise input invariance.
    mean=np.sort(ordered[:,-24:].reshape(-1,4,6),axis=1).mean(axis=1,keepdims=True)
    x[:,-24:]=np.tile(mean,(1,4,1)).reshape(-1,24)
    return x
