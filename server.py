import grpc
from concurrent import futures
import random
import cv2
import rover_pb2
import rover_pb2_grpc


class RoverSensorServicer(rover_pb2_grpc.RoverSensorServicer):
    def __init__(self):
        # Open the default laptop webcam
        self.cap = cv2.VideoCapture(0)

    def GetSensorData(self, request, context):
        # 1. Generate 3 random ultrasonic numbers (e.g., between 5cm and 100cm)
        # Gives a 20% chance to output -1 (infinite), otherwise a random distance between 5 and 100
        ultrasonic_data = [-1 if random.random() < 0.2 else random.randint(5, 100) for _ in range(3)]

        # 2. Capture webcam image
        ret, frame = self.cap.read()
        if not ret:
            image_bytes = b''  # Fallback if camera fails
        else:
            # Encode as JPG to heavily compress the byte stream
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()

        # 3. Return the data matching the .proto schema
        return rover_pb2.SensorData(
            ultrasonic=ultrasonic_data,
            image=image_bytes
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rover_pb2_grpc.add_RoverSensorServicer_to_server(RoverSensorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Test Server running on port 50051...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        print("\nServer stopped.")


if __name__ == '__main__':
    serve()