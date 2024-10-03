import mss
import jax.numpy as jnp
from PIL import Image
import numpy as np

sct = mss.mss()


monitor = sct.monitors[0]
sct_img = sct.grab(monitor)

im = jnp.array(sct_img)

# print(im)
im_drop = jnp.delete(im, 3, axis=2)
print(im_drop)
# img = Image.fromarray(sct_img)

img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
print(sct_img.pixels)
img.save("./test.jpg")

# print(im_drop.shape)