import numpy as np
import torch
import nvdiffrast.torch as dr

# temporary config vars
IMG_DIM = 256


class GenPuzzle:
    def __init__(self, device="cuda"):
        self.device = device

    def init_renderer(self):
        self.glctx = dr.RasterizeCudaContext()

        # initialize modelview and projection matrices (and add batch dimension)
        self.mv = torch.eye(4, device=self.device)[None, ...]
        self.proj = torch.eye(4, device=self.device)[None, ...]

