from PIL import Image, ImageDraw
import mss
import jax.numpy as jnp
sct = mss.mss()

def edge_colours(n_hor_zones, n_vert_zones, screen_side="top"):
    """
    Divides the top or the bottom edges of the screen (by default the top) into a provided number of "zones".
    
    Then returns the average RGB colour in each of those zones.
    
    Output is always going to be an RGB array
    
    :param n_hor_zones The number of horizontal zones there should be.
    :param n_vert_zones The number of vertical zones there should be.
    :param screen_side Specify top or bottom.
    """
        
    for _, monitor in enumerate(sct.monitors[1:], 1):

        sct_img = sct.grab(monitor)
        
        # width of the screen in pixels
        width = sct_img.size.width
        
        # height of the screen in pixels
        height = sct_img.size.height                            
        
        # height of each zone
        zone_height = int(height / n_hor_zones)                 
        
        # width of each zone
        zone_width = int(width / n_vert_zones)                  
        
        # Convert screen into JAX array and determine the truncation range.     
        if screen_side == "top":
            img = jnp.array(sct_img)
            trunc_range = slice(0, zone_height)
            reshape_struct = (n_vert_zones, zone_width, 4)
            
        elif screen_side == "bottom":
            img = jnp.array(sct_img)
            trunc_range = slice(height - zone_height, height - 1)
            reshape_struct = (n_vert_zones, zone_width, 4)
            
        elif screen_side == "left":
            # Convert entire screen as JAX array and swap the height and the width channels while keeping the colour channel the same.
            img = jnp.array(sct_img)
            img = jnp.swapaxes(img, 1, 0)
            trunc_range = slice(0, zone_width)
            reshape_struct = (n_hor_zones, zone_height, 4)
            
        elif screen_side == "right":
            img = jnp.array(sct_img)
            img = jnp.swapaxes(img, 1, 0)
            trunc_range = slice(width - zone_width, width - 1)
            reshape_struct = (n_hor_zones, zone_height, 4)
            
        else: raise ValueError("screen_side must be either top, bottom, left, or right")
        
        img_trunc = img[trunc_range]
        
        # Sum all the pixels on each line across each of the colour channels
        img_line_avg = jnp.average(img_trunc, (0))
        
        # Reshape the image matrix into 4 zones.
        img_reshaped = jnp.reshape(img_line_avg, reshape_struct)
        
        # Get the average colour in each zone.
        img_zone_avg = jnp.average(img_reshaped, (1))
        
        # Convert colour from BRGA to RGB
        return _convert_brga_array_to_rgb_array(img_zone_avg)
    
def _convert_brga_array_to_rgb_array(jax_ar):
    """
    Converts BGRA array and returns an RGB array
    """
    out = []
    
    for each in jax_ar:
        print(each)
        col = [int(each[1]), int(each[2]), int(each[0])]
        out.append(col)
        
    return out

if __name__ == "__main__":
    edge_colours(4, 8, "left")