import serial

if __name__ == "__main__":
    ser = serial.Serial("/dev/ttyACM0", baudrate=9600)
    # while True:
    frame_1 = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0]]
    frame = "\x00\x00ÿ"
    ser.write((frame * 14 + ";").encode("latin-1"))
    