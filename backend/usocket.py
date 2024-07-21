import socket
import time
import os

server_addr = "/tmp/jr.sock"

class USocket:
    """
    Initialises a Unix socket server and starts listening for incoming connections before returning. 
    
    Consider starting up this server in a separate thread since initialising this class will block the thread until a connection is received.
    """
    
    def __init__(self, server_address=server_addr, verbose=False) -> None:
        self.verbose = verbose
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(server_address)
        self.server.listen()
        if self.verbose: print("listening for incoming signals")
        self.server, _ = self.server.accept()
        if self.verbose: print("received connection")
        
    
    def send_message(self, data: list):
        "Sends a message to all connected clients"
        data = str(data).encode()
        if self.verbose: print("Sending message to all clients")
        self.server.sendall(data)
        if self.verbose: print("Message sent (to all clients)")
        
        
    def wait_for_message(self, callback):
        "Freezes the thread and waits until a message is returned from the client"
        self.server, _ = self.server.accept()
        callback()
        pass
    
    
    def close_connection(self):
        "Closes the socket."
        self.server.close()
        os.unlink(server_addr)
        if self.verbose: print("Closed server")
        
        
    def echo(self):
        # self.server.listen()
        # self.server, _ = self.server.accept()
        data = str(self.server.recv(1024).decode("utf-8"))
        self.server.sendall(data.encode())
        
        
        
        
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
        