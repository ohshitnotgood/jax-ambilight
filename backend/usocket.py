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
        # Check if the socket address has already been created because of a previous run
        if os.path.exists(server_address): os.remove(path=server_address)
        self.verbose = verbose
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(server_address)
        self.server.listen()
        self.client_connected = False
        
        
    def wait_for_client(self):
        if self.verbose: print("listening for incoming signals")
        self._server, _ = self.server.accept()
        self.client_connected = True
        if self.verbose: print("received connection")
    
    def send_message(self, data: list):
        "Sends a message to all connected clients"
        data = str(data).encode()
        if self.verbose: print("Sending message to all clients")
        self._server.sendall(data)
        if self.verbose: print("Message sent (to all clients)")
    
    
    def kill_server(self):
        "Kills the server, making it unable to accept new requests."
        self._server.close()
        self.server.close()
        os.remove("/tmp/jr.sock")
    
    def close_connection(self):
        "Closes a socket connection making server available to receive another client."
        self._server.close()
        self.client_connected = False
        if self.verbose: print("Client socket closed")
        
    def wait_for_incoming_msg(self) -> str:
        "Returns any messages that may have been received"
        return str(self._server.recv(1024).decode("utf-8"))
        
        
        
        
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
        