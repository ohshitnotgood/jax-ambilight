import argparse, serial
from usocket import USocket
from cc_colour import c_colours

parser = argparse.ArgumentParser(prog="jax_amb_bnd", description="Background task for jax-ambilight")
parser.add_argument("-v", "--verbose", action="store_true", help="Prints verbose log messages")
parser.add_argument("-d", "--device", nargs='?', const=1, type=str, default="/dev/ttyACM0", help="Specify a device location")
parser.add_argument("-b", "--baudrate", nargs='?', const=1, type=int, default=9600, help="Specify a baudrate for serial communication with the Arduino")



class MainController:
    def __init__(self) -> None:
        self.verbose = parser.parse_args().verbose
        self.preview = parser.parse_args().preview
        self.server = USocket(verbose=self.verbose)
        self.n_height_zones = 4
        self.n_width_zones = 8
        
    def main_loop(self):
        self.server.wait_for_client()
        while True:
            try:
                msg = self.server.wait_for_incoming_msg()
                if self.verbose: print(f"Received message {msg}")
                if msg == "kill_srvr":
                    self.server.send_message("kill_srvr")
                    self.server.close_connection()
                    self.server.wait_for_client()
                elif msg[0:6] == "chg_v:":
                    self.update_zones(msg=msg)
                elif msg == "1001":
                    self.server.send_message("1001")
                    self.server.kill_server()
                    break
                elif msg == "ack_ok":
                    screen_colours = c_colours(n_height_zones=4, n_width_zones=8)
                    self.server.send_message(str(screen_colours).encode())
                else:
                    self.server.send_message(msg)
            except BrokenPipeError:
                self.server.close_connection()
                self.server.wait_for_client()
            except KeyboardInterrupt:
                self.server.kill_server()
                break
        
    def update_zones(self, msg):
        msg = msg[6:]
        msg = msg.split(";")
        self.n_height_zones = msg[0]
        self.n_width_zones = msg[1]
        if self.verbose: print(f"Changed zone: hxw: {self.n_height_zones}x{self.n_width_zones}")
    

class SerialController:
    def __init__(self, verbose, device, baud_rate):
        self.verbose = verbose
        self.ser = serial.Serial(port=device, baudrate=baud_rate)
    
    def main_loop(self):
        while True:
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
            
            self.ser.write((frame + "\n").encode())
            while self.ser.in_waiting:
                ack = self.ser.read_until(size=3).decode()
                if ack != "ack":
                    print("An error occurred writing to the microcontroller.")
                else: print("Receiving acknowledgement sending another frame")

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