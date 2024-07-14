from colour import edge_colours
from usocket import USocket

def main():
    sock_server = USocket()
    try:
        while True:
            top = edge_colours(8, 4, "top")
            sock_server.send_message(top)
    except:
        sock_server.close_connection()

if __name__ == "__main__":
    main()