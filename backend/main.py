import argparse
from usocket import USocket
from cc_colour import c_colours

parser = argparse.ArgumentParser(prog="jax_amb_bnd", description="Background task for jax-ambilight")
parser.add_argument("-v", "--verbose", action="store_true")


class MainController:
    def __init__(self) -> None:
        self.verbose = False
        self.verbose = parser.parse_args().verbose
        self.server = USocket(verbose=self.verbose)
        
    def message_loop(self):
        msg = self.server.wait_for_incoming_msg()
        print(f"Received message {msg}")
        if msg == "kill_srvr":
            self.server.send_message("kill_srvr")
            self.server.close_connection()
        elif msg == "ack_ok":
            screen_colours = c_colours(n_height_zones=4, n_width_zones=8)
            self.server.send_message(str(screen_colours).encode())
            self.message_loop()
        else:
            self.server.send_message(msg)
            self.message_loop()
        

if __name__ == "__main__":
    mc = MainController()
    mc.message_loop()