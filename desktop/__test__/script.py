"""Create a basic Unix socket server that jest tests can communicate to"""
import socket
import os

path = "/tmp/jr.sock"

# Create a socket and bind to address
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path)
srv.listen()

print("listening for incoming messages")

a, b = srv.accept()
a.send("hello world".encode("utf-8"))
print("Send response to the client")


srv.close()
a.close()
os.remove("/tmp/jr.sock")