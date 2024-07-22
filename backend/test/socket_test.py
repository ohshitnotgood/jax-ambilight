"Unit test to test message passing using IPC"

from usocket import USocket
import unittest
import socket
import threading


srvr_addr = "/tmp/jr.sock"

def start_server_for_testing():
    server = USocket()
    server.echo()
    server.close_connection()

class ClientTestClass(unittest.TestCase):
    def setUp(self) -> None:
        thr = threading.Thread(target=start_server_for_testing, args=[])
        thr.start()
        return super().setUp() 
    
    def test_create_client_and_exchange_messages(self):
        # Create and connect to socket
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(srvr_addr)
        
        # Send message and expect the same message
        client.send("hello world".encode("utf-8"))
        msg = client.recv(1024).decode("utf-8")
        self.assertEqual(msg, "hello world")
    
    
if __name__ == "__main__":
    unittest.main()