import mss, platform, platform, jax
import torch.nn.functional as F
import numpy as onp
import torch

sct = mss.mss()
n_colour_channels = 3

def c_colours(n_height_zones: int, n_width_zones: int, inp_img=None, monitor_nr=0, in_colour_space="RGB"):    
    if inp_img == None:
        monitor = sct.monitors[monitor_nr]
        sct_img = sct.grab(monitor)
            
        # width of the screen in pixels
        width = sct_img.size.width
            
        # height of the screen in pixels
        height = sct_img.size.height      
        
        # height of each zone
        zone_height = int(height / n_height_zones)                 
        
        # width of each zone
        zone_width = int(width / n_width_zones)    
        
        img_brga = __array(sct_img)
        
        img = img_brga[:,:,:3]
    else: 
        img = inp_img
        height = img.shape[0]
        width = img.shape[1]
        zone_height = int(height / n_height_zones) 
        zone_width = int(width / n_width_zones) 
    
    # Truncate the image for each sections of the screen
    
    img_trunc_top = img[slice(0, 50)]
    img_trunc_bottom = img[slice(height - 50, height)]
    
    img_swapped = __swapaxes(img, 1, 0)
    
    img_trunc_left = img_swapped[slice(0, 50)]
    img_trunc_right = img_swapped[slice(width - 50, width)]
    
    # Square every value
    img_sqr_top = __square(img_trunc_top)
    img_sqr_bottom = __square(img_trunc_bottom)
    img_sqr_left = __square(img_trunc_left)
    img_sqr_right = __square(img_trunc_right)
    
    
    # Average across each section
    img_line_avg_top = __average(img_sqr_top, (0))
    img_line_avg_bottom = __average(img_sqr_bottom, (0))
    img_line_avg_left = __average(img_sqr_left, (0))
    img_line_avg_right = __average(img_sqr_right, (0))
    
    
    # Reshape the image matrix into 4 zones.
    img_reshaped_top = __reshape(img_line_avg_top, (n_width_zones, zone_width, n_colour_channels))
    img_reshaped_bottom = __reshape(img_line_avg_bottom, (n_width_zones, zone_width, n_colour_channels))
    img_reshaped_left = __reshape(img_line_avg_left, (n_height_zones, zone_height, n_colour_channels))
    img_reshaped_right = __reshape(img_line_avg_right, (n_height_zones, zone_height, n_colour_channels))
    
    
    # Get the average colour in each zone.
    img_zone_avg_top = __average(img_reshaped_top, (1))
    img_zone_avg_bottom = __average(img_reshaped_bottom, (1))
    img_zone_avg_left = __average(img_reshaped_left, (1))
    img_zone_avg_right = __average(img_reshaped_right, (1))
    
    img_zone_sqrt_top = __sqrt(img_zone_avg_top)
    img_zone_sqrt_bottom = __sqrt(img_zone_avg_bottom)
    img_zone_sqrt_left = __sqrt(img_zone_avg_left)
    img_zone_sqrt_right = __sqrt(img_zone_avg_right)
    
    # Convert the BGRA array into desired array type
    rgb_top = _convert_brga_array_to_rgb_array(img_zone_sqrt_top, in_color_space=in_colour_space)
    rgb_bottom = _convert_brga_array_to_rgb_array(img_zone_sqrt_bottom, in_color_space=in_colour_space)
    rgb_left = _convert_brga_array_to_rgb_array(img_zone_sqrt_left, in_color_space=in_colour_space)
    rgb_right = _convert_brga_array_to_rgb_array(img_zone_sqrt_right, in_color_space=in_colour_space)
    
    # for i, each in enumerate(rgb_top):
    #     rgb_top[i] = __get_closest_matching_color(each)
        
        
    # for i, each in enumerate(rgb_bottom):
    #     rgb_bottom[i] = __get_closest_matching_color(each)
        
        
    # for i, each in enumerate(rgb_left):
    #     rgb_left[i] = __get_closest_matching_color(each)
        
        
    # for i, each in enumerate(rgb_right):
    #     rgb_right[i] = __get_closest_matching_color(each)
    
    
    return [rgb_top, rgb_bottom, rgb_left, rgb_right]
    
def __average(inp, axis):
    return torch.mean(inp, axis)
    
def __reshape(inp, new_shape):
    return torch.reshape(inp, new_shape)

def __swapaxes(inp, axis1, axis2):
    return torch.swapaxes(inp, axis1, axis2)
    
def __array(inp):
    """
    Converts the Screenshot object into a numpy array.
    
    On Linux, the numpy array is converted to a JAX array.
    
    On Windows, the numpy array is converted to a torch tensor.
    """
    npar = onp.array(inp)
    return torch.tensor(npar, requires_grad=False, dtype=torch.float16)
    
def __square(inp):
    return inp

def __sqrt(inp):
    return inp

def _convert_brga_array_to_rgb_array(jax_ar, in_color_space="RGB"):
    """
    Converts BGRA array and returns either an RGB or BRG array.
    """
    out = []
    
    for each in jax_ar:
        if in_color_space == "RGB": col = [int(each[2]), int(each[1]), int(each[0])]
        else: col = [int(each[0]), int(each[1]), int(each[2])]
        out.append(col)
        
    return out


def __eucledian_distance(base, jax_ar):
    out = []
    for each in jax_ar:
        out.append(torch.norm(each - base).item())
    return out


def __colours():
    return [
        [5, 5, 173], [8, 8, 105], [3, 3, 77], [64, 64, 201], [44, 44, 222], [99, 99, 219], [133, 133, 222],         # shades of blue
        [65, 184, 61], [41, 181, 36], [21, 171, 15], [10, 133, 5], [5, 99, 2], [0, 28, 0], [18, 153, 14],           # shades of green
        [153, 146, 14], [209, 199, 9], [255, 242, 0], [255, 221, 0], [212, 187, 23], [143, 125, 9], [184, 161, 13], # shades of yellow
        [184, 90, 13], [222, 103, 7], [255, 115, 0], [168, 108, 12],                                                # shades of orange
        [36, 71, 117], [41, 116, 214], [2, 68, 156], [42, 7, 107], [61, 101, 153], [18, 62, 120], [66, 135, 245],   # more shades of blue
        [89, 18, 120], [137, 10, 91], [211,3, 252], [140, 12, 166], [152, 66, 245], [62, 1, 128], [122, 25, 173], [127, 43, 217], [52, 0, 128],              # shades of purple
        [128, 0, 108], [192, 2, 167], [219, 53, 194], [252, 40, 221], [252, 0, 217],                                # shades of magenta
        [255, 0, 146], [184, 0, 98], [133, 1, 71], [237, 33, 142], [158, 35, 101], [171, 14, 97], [255, 0, 149], [184, 0, 107], [184, 0, 165],                   # shades of pink
        [153, 171, 14], [193, 214, 36], [106, 120, 5], [162, 184, 0], [169, 186, 35], [103, 114, 14],               # shades of lime
        [212, 13, 13], [255, 0, 0], [176, 18, 18], [191, 0, 0], [168, 0, 0], [255, 38, 48],                         # shades of red
        [38, 255, 241], [41, 204, 193], [0, 166, 155], [0, 255, 255], [27, 135, 140],                               # shades of teal
        [238, 248, 0], [207, 214, 0], [156, 161, 8],                                                                 # shades of yellow
        [0, 12, 21], [12, 12, 12], [12, 0, 12], [12, 0, 12], [22, 12, 3], [1, 1, 1], [10, 10, 10], [20, 20, 20]
    ]

def __get_closest_matching_color(base):
    eucd = __eucledian_distance(torch.tensor(base, dtype=torch.float16), torch.tensor(__colours(), dtype=torch.float16))
    sm = torch.nn.functional.softmin(torch.tensor(eucd))
    r = 0
    g = 0
    b = 0
    for i, each in enumerate(sm):
        r += each * __colours()[i][0]
        g += each * __colours()[i][1]
        b += each * __colours()[i][2]
        
    return [int(r), int(g), int(b)]


if __name__ == "__main__":
    print(c_colours(3, 4))