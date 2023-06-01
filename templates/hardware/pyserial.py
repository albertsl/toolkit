import serial

if __name__ == '__main__':
    port = 'COM3'
    baud_rate = 115200

    # Open the serial port
    ser = serial.Serial(port, baud_rate)

    # Check if the serial port is open
    if ser.is_open:
        print(f"Serial port {port} is open.")

        # Get data from the serial port
        print(ser.readline())

        # Send data in byte representation
        data = b'\x2D'
        ser.write(data)

        # Send data in plain text
        data = '-' # Data to be sent
        ser.write(data.encode()) # Convert data to bytes and send it

        # Close the serial port
        ser.close()
        print("Serial port closed.")
    else:
        print(f"Failed to open serial port {port}.")