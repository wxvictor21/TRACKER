import serial
import time

class GRBLController:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def send(self, cmd):
        """Sends a command to GRBL and waits for an 'ok' or 'error' response."""
        print(f"Sending to GRBL: {cmd}")
        self.ser.reset_input_buffer()
        self.ser.write((cmd + '\n').encode())
        
        response_lines = []
        t0 = time.time()
        while time.time() - t0 < 2.0: # 2 second timeout
            line = self.ser.readline().decode().strip()
            if line:
                print(f"GRBL response: {line}")
                response_lines.append(line)
                if 'ok' in line or 'error' in line:
                    break
        
        if not response_lines:
            print("GRBL did not respond.")

        return response_lines

    def move(self, x_units, y_units, f=1500):
        """Convierte unidades en mm y envía comando de movimiento"""
        x_mm = float(x_units) / 50.0   # 1 unidad en X = 50 mm
        y_mm = float(y_units) / 10.0   # 1 unidad en Y = 10 mm
        cmd = f'G1 X{x_mm:.2f} Y{y_mm:.2f} F{f}'
        resp = self.send(cmd)
        return cmd, resp

    def status(self):
        return self.send('?')
