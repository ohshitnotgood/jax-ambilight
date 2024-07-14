"""
Basic Unix socket communication testing
"""

import socket

# Create simple socket client
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)


# Connect client to the server
client.connect("/tmp/ssctdddd.sock")

# Print out every message that is received from the server.
while True:
    r = client.recv(1000)
    print(r.decode())