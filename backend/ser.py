import serial, argparse, time
from cc_colour import c_colours

parser = argparse.ArgumentParser(prog="jax_amb_bnd", description="Background task for jax-ambilight")
parser.add_argument("-v", "--verbose", action="store_true", help="Prints verbose log messages")
parser.add_argument("-d", "--device", nargs='?', const=1, type=str, default="/dev/ttyACM0", help="Specify a device location")
parser.add_argument("-b", "--baudrate", nargs='?', const=1, type=int, default=9600, help="Specify a baudrate for serial communication with the Arduino")


class SerialController:
    def __init__(self, verbose, device, baud_rate):
        self.verbose = verbose
        self.ser = serial.Serial(port=device, baudrate=baud_rate)
    
    def main_loop(self):
        while True:
            ack = ""
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
            
            self.ser.write((frame + ";").encode())
            # while ack != "ack":
            #     ack = self.ser.read_until(size=3).decode()
                
            print(frame, "sending another frame")
            time.sleep(1)
            

if __name__ == "__main__":
    parser.parse_args()
    verbose = parser.parse_args().verbose
    device = parser.parse_args().device
    baudrate = parser.parse_args().baudrate
    ser_con = SerialController(verbose=verbose, device=device, baud_rate=baudrate)
    ser_con.main_loop()
    # mc = MainController()
    # mc.main_loop()
    pass