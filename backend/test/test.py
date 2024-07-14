from PIL import Image, ImageDraw
from colour import edge_colours
import time

def draw(c):
    out = Image.new("RGB", (120, 30), (255, 255, 255))
    one = [0, 0, 30, 30]
    two = [30, 0, 60, 30]
    three = [60, 0, 90, 30]
    four = (90, 0, 120, 30)
    
    imgd = ImageDraw.Draw(out)
    imgd.rectangle(one, fill=tuple(c[0]))
    imgd.rectangle(two, fill=tuple(c[1]))
    imgd.rectangle(three, fill=tuple(c[2]))
    imgd.rectangle(four, fill=tuple(c[3]))
    out.show()
    pass
        
        
if __name__ == "__main__":
    time.sleep(2)
    co = edge_colours(8, 4, "bottom")
    draw(co)
    