import serial, argparse, time, platform
from cc_colour import c_colours
parser = argparse.ArgumentParser(prog="jax_amb_bnd", description="Background task for jax-ambilight")
parser.add_argument("-v", "--verbose", action="store_true", help="Prints verbose log messages")
parser.add_argument("-d", "--device", nargs='?', const=1, type=str, default="/dev/ttyACM0" if platform.system() == "Linux" else "COM5", help="Specify a device location")
parser.add_argument("-b", "--baudrate", nargs='?', const=1, type=int, default=9600, help="Specify a baudrate for serial communication with the Arduino")


parser.parse_args()
verbose = parser.parse_args().verbose
device = parser.parse_args().device
baudrate = parser.parse_args().baudrate


class SerialController:
    def __init__(self, verbose, device, baud_rate):
        self.verbose = verbose
        self.ser = serial.Serial(port=device, baudrate=baud_rate)
    
    def main_loop(self):
        frame_rate = 0
        
        while True:
            start = time.time()
            ack = ""
            [top, bottom, left, right] = c_colours(3, 4)
            frame = ""
            for each in top:
                for each_subpixel in each:
                    if each_subpixel == 59:
                        each_subpixel = 58
                    frame += chr(each_subpixel)
            for each in bottom:
                for each_subpixel in each:
                    if each_subpixel == 59:
                        each_subpixel = 58
                    frame += chr(each_subpixel)
            for each in left:
                for each_subpixel in each:
                    if each_subpixel == 59:
                        each_subpixel = 58
                    frame += chr(each_subpixel)
            for each in right:
                for each_subpixel in each:
                    if each_subpixel == 59:
                        each_subpixel = 58
                    frame += chr(each_subpixel)
            
            self.ser.write((frame + ";").encode("latin-1"))
            while ack != "ack":
                ack = self.ser.read_until(size=3).decode()
                
            end = time.time()
            frame_rate = 1 / (end - start)
            
            print(str(frame_rate), end="\r", flush=True)
            

if __name__ == "__main__":
    ser_con = SerialController(verbose=verbose, device=device, baud_rate=baudrate)
    ser_con.main_loop()