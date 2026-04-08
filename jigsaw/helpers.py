import numpy as np
from PIL import Image
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def save_mesh(fname, verts, faces, face_type="triangle"):
    mesh = meshio.Mesh(
        points = verts,
        cells = [(face_type, faces)]
    )
    mesh.write(fname)


def read_mesh(fname, face_type="triangle"):
    mesh = meshio.read(fname)
    Ps = mesh.points
    Es = mesh.cells_dict[face_type]
    return Ps, Es

def read_image(fname, w=None, h=None, format="L", resample=Image.Resampling.BILINEAR):
    with Image.open(fname) as pimg:
        # img = np.asarray(pimg.convert("RGB")) / 255
        if w and h:
            img = np.asarray(pimg.resize((w, h), resample=resample).convert(format)) / 255
        else:
            img = np.asarray(pimg.convert(format)) / 255
    return img

def save_images(fname, imgs, grid_size=None, auto_grid=False, dpi=200, show_ax="off", show_cbar=True, **kwargs):
    if grid_size is None or grid_size[0] * grid_size[1] < len(imgs):
        grid_size = (len(imgs), 1)
    if auto_grid:
        w = h = int(np.sqrt(len(imgs)))
        while w * h < len(imgs):
            h += 1
        grid_size = (w, h)
        # grid_size = 
    fig, axs = plt.subplots(nrows=grid_size[0], ncols=grid_size[1], dpi=dpi)
    plt.rc('font', size=6)
    if len(imgs) == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    for idx, a in enumerate(axs):
        if idx < len(imgs):
            i = imgs[idx]
            if i.ndim == 2 or i.shape[2] == 1:
                im = a.imshow(i, cmap="gray", **kwargs)
                divider = make_axes_locatable(a)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                if show_cbar:
                    fig.colorbar(im, cax=cax)
            else:
                a.imshow(i, **kwargs)
        a.axis(show_ax)
    fig.savefig(fname)
    plt.close(fig)

def read_token(tkn_file):
    with open(tkn_file, "r") as f:
        return f.readline().strip()
    
class FigureGrid:
    def __init__(self, grid_size, off=True, **kwargs):
        if type(grid_size) == int:
            grid_size = (grid_size, grid_size)
        self.fig, self.axs = plt.subplots(grid_size[0], grid_size[1], **kwargs)
        self.fig.tight_layout()
        self.axs = self.axs.flatten()
        self.i = 0

        if off:
            [ax.axis('off') for ax in self.axs]

    def save(self, fname):
        self.fig.savefig(fname)

    def close(self):
        plt.close(self.fig)

    def next(self):
        if self.i >= len(self.axs):
            raise IndexError("FigureGrid is full")
        ax = self.axs[self.i]
        self.i += 1
        return ax

    def reset(self):
        self.i = 0
    
    def get(self, i):
        if i >= len(self.axs):
            raise IndexError("FigureGrid index out of range")
        return self.axs[i]
