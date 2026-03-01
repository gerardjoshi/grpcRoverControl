import grpc
import time
import cv2
import numpy as np
import ollama
import json
import rover_pb2
import rover_pb2_grpc


# --- 1. Output Actions ---
def move_rover(direction: str):
    print(f"[ACTION EXECUTED] Motor moving: {direction}")


def drop_payload():
    print(f"[ACTION EXECUTED] Payload Dropped!")


# --- 2. Deterministic CV Setup ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def detect_aruco(image_bytes: bytes) -> bool:
    """Instantly checks for ArUco markers before the LLM processes the image."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    if frame is None: return False
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    return ids is not None and len(ids) > 0


# --- 3. The Orchestration Loop ---
def run_agent():
    channel = grpc.insecure_channel('localhost:50051')
    stub = rover_pb2_grpc.RoverSensorStub(channel)
    model_name = 'llama3.2-vision'

    print(f"Initializing JSON-Structured Vision Loop with {model_name}...")

    while True:
        try:
            # A. Sense (gRPC)
            response = stub.GetSensorData(rover_pb2.EmptyRequest())
            ultrasonic = list(response.ultrasonic)

            # Save frame for Ollama
            image_path = 'current_view.jpg'
            with open(image_path, 'wb') as f:
                f.write(response.image)

            # B. Pre-process (Fast OpenCV)
            start_cv = time.time()
            aruco_detected = detect_aruco(response.image)
            cv_time = (time.time() - start_cv) * 1000

            # C. Think (VLM + Context via JSON)
            # We explicitly ask for a JSON payload instead of using the tools parameter
            prompt = (
                "You are the brain of an autonomous rover.\n"
                f"1. Hardware ArUco Sensor: {'DETECTED' if aruco_detected else 'NONE'}\n"
                f"2. Ultrasonic Array (cm) [Front, Front-Left, Front-Right]: {ultrasonic}\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- If the Hardware ArUco Sensor is DETECTED, you MUST execute 'drop_payload'.\n"
                "- Look closely at the provided image. If you see a physical obstacle (like a chair, box, or wall) blocking the path that the ultrasonic sensors missed, execute 'move_rover' to avoid it.\n"
                "- If any ultrasonic sensor is < 30cm, execute 'move_rover' to avoid it.\n"
                "- If the path is visually clear and sensors are > 30cm, execute 'move_rover' front.\n\n"
                "You MUST respond ONLY with a valid JSON object in one of these exact formats:\n"
                '{"function": "move_rover", "direction": "<direction>"}\n'
                'OR\n'
                '{"function": "drop_payload"}'
            )

            print(f"\n[{time.strftime('%X')}] ArUco: {aruco_detected} ({cv_time:.1f}ms). VLM analyzing scene...")

            res = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
                format='json'  # This forces the model to guarantee valid JSON syntax
            )

            # D. Act (Manual JSON Parsing)
            try:
                raw_response = res.get('message', {}).get('content', '{}')
                decision = json.loads(raw_response)

                func_name = decision.get('function')

                if func_name == 'move_rover':
                    direction = decision.get('direction', 'front')
                    move_rover(direction)
                elif func_name == 'drop_payload':
                    drop_payload()
                else:
                    print(f"[WARNING] Model hallucinated an unknown function: {func_name}")

            except json.JSONDecodeError:
                print(f"[WARNING] Model failed to output valid JSON. Raw output: {raw_response}")

        except grpc.RpcError:
            print("[ERROR] gRPC disconnected. Retrying...")
            time.sleep(2)


if __name__ == '__main__':
    run_agent()