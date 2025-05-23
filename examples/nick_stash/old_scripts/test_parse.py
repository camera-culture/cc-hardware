import serial
# pip install pySerial

PORT = "/dev/cu.usbmodem1103"
BAUDRATE = 921600
ser = serial.Serial(
    port=PORT,\
    baudrate=921600,\
    parity=serial.PARITY_NONE,\
    stopbits=serial.STOPBITS_ONE,\
    bytesize=serial.EIGHTBITS,\
        timeout=0)

print("connected to: " + ser.portstr)

#this will store the line
# line = []

class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)


# ser = serial.Serial('COM7', 9600)
rl = ReadLine(ser)

count_lines = 0
max_lines = 40
# while True:
while count_lines < max_lines:
    print("reading line")
    line_bytearr = rl.readline()
    line_str = line_bytearr.decode("utf-8")
    # print(line_bytearr)
    # print(type(line_bytearr))
    print(repr(line_str))
    count_lines += 1

ser.close()