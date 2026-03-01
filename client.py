import grpc
import time
import cv2
import numpy as np
import rover_pb2
import rover_pb2_grpc


def callInputs(stub, history):
    """Fetches data via gRPC, decodes the image, and logs to history."""
    # 1. Trigger the gRPC request
    response = stub.GetSensorData(rover_pb2.EmptyRequest())
    timestamp = time.time()

    # 2. Decode the raw bytes into an OpenCV image matrix
    image_array = np.frombuffer(response.image, dtype=np.uint8)
    decoded_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    ultrasonic_data = list(response.ultrasonic)

    # 3. Store the raw data in the dictionary
    history[timestamp] = {
        "ultrasonic": ultrasonic_data,
        "image_data": response.image  # Storing raw bytes is more memory efficient
    }

    return decoded_frame, ultrasonic_data


def run():
    # Connect to the local test server
    channel = grpc.insecure_channel('localhost:50051')
    stub = rover_pb2_grpc.RoverSensorStub(channel)
    history = {}

    window_name = "Rover Client Interface"
    print("Opening interface... Please ensure the OpenCV window is clicked/focused.")
    print("Press the 'SPACEBAR' to fetch data.")
    print("Press 'q' to quit.")

    # Create an initial blank window to capture keypresses
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Press SPACE to fetch data", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, placeholder)

    try:
        while True:
            # Wait 10ms for a keypress. & 0xFF ensures cross-platform mapping.
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar pressed
                frame, sensors = callInputs(stub, history)

                if frame is not None:
                    # Overlay the sensor data directly onto the video frame for visual confirmation
                    cv2.putText(frame, f"Sensors: {sensors}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name, frame)
                    print(f"[{time.time()}] Data captured -> Sensors: {sensors}")

    except KeyboardInterrupt:
        print("\nClient stopped manually.")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run()