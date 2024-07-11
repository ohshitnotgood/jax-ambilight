from PIL import Image, ImageDraw
import mss
import jax.numpy as jnp
sct = mss.mss()

def top_zone_colours(n_hor_zones, n_vert_zones):
    """
    Returns the average colours in each of the zones that are on the top of the screen
    
    Output is always going to be an RGB array
    
    TODO: fix reshaping errors for different screen sizes
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
        
        # entire screen as a JAX array
        img = jnp.array(sct_img)          
                     
        
        # Only zones that are on the top of the screen.
        # img_trunc should now have the shape (zone_height, width, 4) where zone_height
        # denotes the number of lines there are in each zone.
        img_trunc = img[0:zone_height]
        
        # Sum all the pixels on each line across each of the colour channels
        img_line_avg = jnp.average(img_trunc, (1))
        
        # Reshape the image matrix into 4 zones.
        img_reshaped = jnp.reshape(img_line_avg, (int(zone_height / n_vert_zones), 4, 4))
        
        # Get the average colour in each zone.
        img_zone_avg = jnp.average(img_reshaped, (1))
        
        return _convert_brga_array_to_rgb_array(img_zone_avg)
    
    
def _convert_brga_array_to_rgb_array(jax_ar):
    """
    Converts BGRA array and returns an RGB array
    """
    out = []
    
    for each in jax_ar:
        col = [int(each[1]), int(each[2]), int(0)]
        out.append(col)
        
    return out