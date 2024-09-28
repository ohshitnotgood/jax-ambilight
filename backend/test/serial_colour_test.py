import serial

if __name__ == "__main__":
    ser = serial.Serial("/dev/ttyACM0", baudrate=9600)
    while True:
        to_write = input()
        ser.write((to_write*14 + ";").encode())
    