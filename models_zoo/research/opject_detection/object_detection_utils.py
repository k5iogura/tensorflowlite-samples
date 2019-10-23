import numpy as np
def softmax(c1):
    classes   = c1.shape[-1]
    cx        = np.max(c1,axis=-1)
    tilespec  = [1]*cx.ndim
    tilespec.append(classes)
    exp_a     = np.exp(c1 - np.tile(cx[...,np.newaxis], tuple(tilespec)))
    sum_exp_a = np.sum(exp_a,axis=-1)
    y = exp_a / np.tile(sum_exp_a[...,np.newaxis], tuple(tilespec))
    return y 
