import numpy as np
import nvdiffrast.torch as dr

import torch
from torch.optim.lr_scheduler import LambdaLR

from ts_simple.sd_guidance import StableDiffusionGuidance, StableDiffusionPromptProcessor
from ts_simple.df_guidance import DeepFloydGuidance, DeepFloydPromptProcessor

# differentiably render V, F using nvdiffrast
# assumes V is in homogeneous coordinates (and batched)
def render(V, F, C, background, glctx, res=256):
    rast, _ = dr.rasterize(glctx, V, F, resolution=[res, res])
    mask = rast[..., -1:] == 0
    col, _ = dr.interpolate(C, rast, F)
    out = dr.antialias(torch.where(mask, background, col), rast, V, F)
    return out

if __name__ == "__main__":
    DEVICE = "cuda:0"
    RES = 256
    ITERATIONS = 1000
    LR = 1e-1

    # we try to optimize a single shape
    # start with a circle
    N = 64

    # 