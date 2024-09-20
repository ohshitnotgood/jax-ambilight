import serial
from cc_colour import c_colours

class SerialController:
    def __init__(self):
        pass
    
    
    def write_msg(self, msg):
        pass
    
    
if __name__ == "__main__":
    # ser = serial.Serial("/dev/ttyACM0", baudrate=9600)
    [top, bottom, left, right] = c_colours(3, 4)
    frame = ""
    
    for each in top:
        for each_subpixel in each:
            frame += f"{each_subpixel:03}"
    for each in bottom:
        for each_subpixel in each:
            frame += f"{each_subpixel:03}"
    for each in left:
        for each_subpixel in each:
            frame += f"{each_subpixel:03}"
    for each in right:
        for each_subpixel in each:
            frame += f"{each_subpixel:03}"
    
    print(frame)
    # ser.write(frame.encode())