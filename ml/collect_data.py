"""
collect_data.py
----------------
Reads the CSV stream from dual_mpu6050_calibrated.ino and saves labeled
gesture recordings to disk for later ML training.

Workflow:
    1. Update SERIAL_PORT below to match Arduino IDE's Tools > Port.
    2. Close the Arduino IDE Serial Monitor first (only one program can
       hold the serial port at a time).
    3. Run: python collect_data.py
    4. Type a gesture name + Enter to start recording. Perform the
       gesture repeatedly for a few seconds.
    5. Press Enter (blank input) to stop that recording.
    6. Repeat for each gesture, including an "idle" recording (no motion)
       -- this matters a lot for a usable classifier.
    7. Ctrl+C to quit.

Each recording session is saved as its own CSV file in ml/data/raw/.
"""

import csv
import threading
import time
import serial

SERIAL_PORT = "COM3"      # <-- change to match your Arduino IDE port
BAUD_RATE = 115200
OUTPUT_DIR = "data/raw"

HEADER = ["timestamp_ms", "label", "aw", "ax", "ay", "az",
          "bw", "bx", "by", "bz"]


class SerialLogger:
    def __init__(self, serial_conn):
        self.serial_conn = serial_conn
        self.writer = None
        self.file = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self.running:
            try:
                line = self.serial_conn.readline().decode(errors="ignore").strip()
            except Exception:
                continue

            if not line or line.startswith("timestamp_ms") or "," not in line:
                continue  # skip header echoes, boot messages, blank lines

            with self.lock:
                if self.writer:
                    row = line.split(",")
                    if len(row) == len(HEADER):
                        self.writer.writerow(row)

    def start_recording(self, filepath):
        with self.lock:
            self.file = open(filepath, "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(HEADER)

    def stop_recording(self):
        with self.lock:
            if self.file:
                self.file.close()
            self.file = None
            self.writer = None

    def stop(self):
        self.running = False


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # allow ESP32 to reset after opening the port
    ser.reset_input_buffer()

    logger = SerialLogger(ser)

    print("Connected. Type a gesture name + Enter to start recording it.")
    print("Blank input + Enter stops recording. Ctrl+C to quit.\n")

    try:
        while True:
            label = input("Gesture name (blank = stop): ").strip()

            if label:
                ser.write(f"START:{label}\n".encode())
                filename = f"{OUTPUT_DIR}/{label}_{int(time.time())}.csv"
                logger.start_recording(filename)
                print(f"Recording '{label}' -> {filename}")
                print("Press Enter to stop this gesture.")
                input()
                ser.write(b"STOP\n")
                logger.stop_recording()
                print("Stopped.\n")

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        logger.stop()
        ser.close()


if __name__ == "__main__":
    main()
