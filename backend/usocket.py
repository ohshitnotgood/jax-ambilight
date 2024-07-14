import socket
import time
import os

server_addr = "/tmp/ssctdddd.sock"

class USocket:
    """
    Initialises a Unix socket server and starts listening for incoming connections before returning. 
    
    Consider starting up this server in a separate thread since initialising this class will block the thread until a connection is received.
    """
    
    def __init__(self) -> None:
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(server_addr)
        self.server.listen()
        print("listening for incoming signals")
        self.server, _ = self.server.accept()
        print("received connection")
        
    
    def send_message(self, data: list):
        "Sends a message to all connected clients"
        data = str(data).encode()
        self.server.sendall(data)
        
        
    def close_connection(self):
        "Closes the socket."
        self.server.close()
        
        
        
def _test_server(server):
    "Once the server is created, sends infinite number of messages to all clients"
    count = [0]
    while True:
        print(f"Sending {count}")
        server.send_message(count)
        count[0] += 1
        
        
        
if __name__ == "__main__":
    server = USocket()
    try: _test_server(server=server)
    except Exception as e:
        print(e)
        server.close_connection()
        os.remove(server_addr)
        