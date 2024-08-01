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
        
    def message_loop(self):
        try:
            while True:
                msg = self.server.wait_for_incoming_msg()
                if self.verbose: print(f"Received message {msg}")
                if msg == "kill_srvr":
                    self.server.send_message("kill_srvr")
                    self.server.close_connection()
                elif msg == "ack_ok":
                    screen_colours = c_colours(n_height_zones=4, n_width_zones=8)
                    self.server.send_message(str(screen_colours).encode())
                else:
                    self.server.send_message(msg)
        except KeyboardInterrupt or BrokenPipeError:
            if self.verbose: print("Encountered KeyboardInterrupt or client disconnected, closing server")
            self.server.close_connection()
        

if __name__ == "__main__":
    mc = MainController()
    mc.main_loop()