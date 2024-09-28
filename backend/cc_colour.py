import mss, platform
import jax.numpy as jnp
import torch

sct = mss.mss()
def c_colours(n_height_zones: int, n_width_zones: int, inp_img=None, monitor_nr=0, in_colour_space="BRGA"):
    if in_colour_space == "RGB": n_colour_channels = 3
    elif in_colour_space == "BRGA": n_colour_channels = 4
    else: raise ValueError("out_colour_space must be one of string values: RGB, RGBA or BRGA")
    
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
        
        img = __array(sct_img)
    else: 
        img = inp_img
        height = img.shape[0]
        width = img.shape[1]
        zone_height = int(height / n_height_zones) 
        zone_width = int(width / n_width_zones) 
    
    # Truncate the image for each sections of the screen
    img_trunc_top = img[slice(0, zone_height)]
    img_trunc_bottom = img[slice(height - zone_height, height - 1)]
    
    img_swapped = __swapaxes(img, 1, 0)
    
    img_trunc_left = img_swapped[slice(0, zone_width)]
    img_trunc_right = img_swapped[slice(width - zone_width, width - 1)]
    
    
    # Average across each section
    img_line_avg_top = __average(img_trunc_top, (0))
    img_line_avg_bottom = __average(img_trunc_bottom, (0))
    img_line_avg_left = __average(img_trunc_left, (0))
    img_line_avg_right = __average(img_trunc_right, (0))
    
    
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
    
    # Convert the BGRA array into desired array type
    rgb_top = _convert_brga_array_to_rgb_array(img_zone_avg_top, in_color_space=in_colour_space)
    rgb_bottom = _convert_brga_array_to_rgb_array(img_zone_avg_bottom, in_color_space=in_colour_space)
    rgb_left = _convert_brga_array_to_rgb_array(img_zone_avg_left, in_color_space=in_colour_space)
    rgb_right = _convert_brga_array_to_rgb_array(img_zone_avg_right, in_color_space=in_colour_space)
    
    return [rgb_top, rgb_bottom, rgb_left, rgb_right]
    
def __average(inp, axis):
    if platform.system() == 'Linux':
        return jnp.average(inp, axis)
    elif platform.system() == 'Windows':
        return torch.mean(inp, axis)
    else: raise OSError("Operating system not supported")
    
def __reshape(inp, new_shape):
    if platform.system() == 'Linux':
        return jnp.reshape(inp, new_shape)
    elif platform.system() == 'Windows':
        return torch.reshape(inp, new_shape)
    else: raise OSError("Operating system not supported")

def __swapaxes(inp, axis1, axis2):
    if platform.system() == 'Linux':
        return jnp.swapaxes(inp, axis1, axis2)
    elif platform.system() == 'Windows':
        return torch.swapaxes(inp, axis1, axis2)
    else: raise OSError("Operating system not supported")
    
def __array(inp):
    if platform.system() == 'Linux':
        return jnp.array(inp)
    elif platform.system() == 'Windows':
        return torch.tensor(inp, requires_grad=False)
    else: raise OSError("Operating system not supported")


def _convert_brga_array_to_rgb_array(jax_ar, in_color_space="RGB"):
    """
    Converts BGRA array and returns either an RGB or BRG array.
    """
    out = []
    
    for each in jax_ar:
        if in_color_space == "RGB": col = [int(each[1]), int(each[2]), int(each[0])]
        else: col = [int(each[0]), int(each[1]), int(each[2])]
        out.append(col)
        
    return out
    
    

if __name__ == "__main__":
    print(c_colours(3, 4))