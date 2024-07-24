from PIL import Image, ImageDraw
import random
import jax.numpy as jnp
from cc_colour import c_colours
import unittest

def create_test_image(width=2560, height=1440, n_width_zones=8, n_height_zones=4, colours=None):
    test_img = Image.new("RGB", (width, height), (255, 255, 255))
    zone_width = width / n_width_zones
    zone_height = height / n_height_zones
    
    imgd = ImageDraw.Draw(test_img)
    
    if colours == None:
        colours = generate_zone_colours(n_width_zones=n_width_zones, n_height_zones=n_height_zones)
    
    for i in range(0, n_width_zones, 1):
        rect_dims = [i * zone_width, 0, (i + 1) * zone_width, zone_height]
        imgd.rectangle(rect_dims, fill=colours[0][i])
        
    
    for i in range(0, n_width_zones, 1):
        rect_dims = [i * zone_width, height - zone_height, (i + 1) * zone_width, height]
        imgd.rectangle(rect_dims, fill=colours[1][i])
        
    
    for i in range(0, n_height_zones, 1):
        rect_dims = [0, i * zone_height, zone_width, (i + 1) * zone_height]
        imgd.rectangle(rect_dims, fill=colours[2][i])
        
    for i in range(0, n_height_zones, 1):
        rect_dims = [width - zone_width, i * zone_height, width, (i + 1) * zone_height]
        imgd.rectangle(rect_dims, fill=colours[3][i])
        
    
    test_img.save("test.png") 
    # return test_img
    

def generate_zone_colours(n_width_zones=8, n_height_zones=4):
    out = []
    a = []
    b = []
    for _ in range(0, n_width_zones):
        a.append((random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
        b.append((random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
        
    out.append(a)
    out.append(b)
    
    a = [] 
    b = []
    for _ in range(0, n_height_zones):
        a.append((random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
        b.append((random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
        
    out.append(a)
    out.append(b)
    
    out[2][0] = out[0][0]
    out[3][0] = out[0][-1]
    
    out[2][-1] = out[1][0]
    out[3][-1] = out[1][-1]
    
    return out
    
    
class ColourTestClass(unittest.TestCase):
    def test_averaging_colours_across_zones(self):    
        n_height_zones = 4
        n_width_zones = 8
        colours = generate_zone_colours()
        test_img = jnp.array(create_test_image(colours=colours))
        out_colours = c_colours(n_height_zones=n_height_zones, n_width_zones=n_width_zones, inp_img=test_img)
        self.assertEqual(test_img, out_colours)


if __name__ == "__main__":
    # unittest.main
    create_test_image()