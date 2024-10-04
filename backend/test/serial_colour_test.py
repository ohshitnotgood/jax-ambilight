import serial

if __name__ == "__main__":
    ser = serial.Serial("/dev/ttyACM0", baudrate=9600)
    c = 0
    while True:
        to_write = input("")
        ser.write((";").encode())
        print(c)
        c += 1
    