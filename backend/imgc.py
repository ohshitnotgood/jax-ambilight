import mss
import jax.numpy as jnp
from PIL import Image
import numpy as np
import torch

sct = mss.mss()


monitor = sct.monitors[0]
sct_img = sct.grab(monitor)

im = torch.tensor(np.array(sct_img))

img = im[:,:,:3]
img = img.flip(2)
img = torch.swapaxes(img, 1, 0)
pilimage = Image.fromarray(np.array(img))
pilimage.save("out.png")
# print(img.shape)