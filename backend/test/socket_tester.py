"""
A test script for testing IPC sockets

Run this test script to connect to the IPC server and send messages to it.

For this particular project, sending ack_ok should return average colours for each of the zones on screen,
sending kill_srvr should end the connection and sending anything else should echo.
"""
import socket

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srvr_addr = str(input("Enter server address (default: /tmp/jr.sock): "))
if srvr_addr == "": srvr_addr = "/tmp/jr.sock"

try:
    print("Connecting to socket")
    client.connect(srvr_addr)
    msg = ""
    
    while msg != "kill_srvr":
        send = str(input("Write a message to send: "))
        client.send(send.encode())
        msg = client.recv(1024).decode()
        print(f"Received message from server: {msg}")
except Exception as e:
    print(e)