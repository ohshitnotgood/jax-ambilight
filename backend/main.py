from cc_colour import c_colours
from usocket import USocket

def main():
    sock_server = USocket()
    try:
        while True:
            colours = c_colours(n_height_zones=4, n_width_zones=8)
            sock_server.send_message(colours)
    except:
        sock_server.close_connection()

if __name__ == "__main__":
    main()