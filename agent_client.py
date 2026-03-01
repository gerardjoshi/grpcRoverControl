import grpc
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Import compiled gRPC files
import rover_pb2
import rover_pb2_grpc


# --- 1. Network Action Wrappers ---
def move_rover(stub, direction: str):
    """Sends movement command over gRPC to the Pi."""
    print(f"[NETWORK] Sending Move Command: {direction.upper()}")
    stub.ExecuteCommand(rover_pb2.CommandRequest(action="move", direction=direction))


def drop_payload(stub):
    """Sends payload drop command over gRPC to the Pi."""
    print(f"[NETWORK] Sending Drop Command!")
    stub.ExecuteCommand(rover_pb2.CommandRequest(action="drop", direction=""))


# --- 2. CV & Sensor Setup ---
print("Loading Computer Vision Models...")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
# Filter out tiny background artifacts to prevent false positives
aruco_params.minMarkerPerimeterRate = 0.08
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

yolo_model = YOLO('yolov8n.pt')

ULTRASONIC_THRESHOLD_CM = 30
MIN_OBJECT_AREA = 15000
TEST_DELAY_SECONDS = 0.5


# --- 3. Sensor Fusion & Decision Matrix ---
def decide_movement(us_data: list, cv_blocked: dict) -> str:
    """
    Fuses Ultrasonic and YOLO Camera data.
    CRITICAL: Arduino formats us_data as [Left, Front, Right]
    """
    # 0. Sanitize Data: Replace -1 (infinite distance) with 9999
    clean_us = [9999 if x == -1 else x for x in us_data]

    # 1. Map Indices: [0] = Left, [1] = Front, [2] = Right
    left_blocked = clean_us[0] < ULTRASONIC_THRESHOLD_CM or cv_blocked["left"]
    center_blocked = clean_us[1] < ULTRASONIC_THRESHOLD_CM or cv_blocked["center"]
    right_blocked = clean_us[2] < ULTRASONIC_THRESHOLD_CM or cv_blocked["right"]

    # 2. Decision Tree
    if center_blocked:
        if left_blocked and right_blocked:
            return "back"
        elif left_blocked and not right_blocked:
            return "right"
        elif right_blocked and not left_blocked:
            return "left"
        else:
            # Turn toward the side with the most physical open space
            return "left" if clean_us[0] > clean_us[2] else "right"
    else:
        if left_blocked and right_blocked:
            return "front"
        elif left_blocked and not right_blocked:
            return "front_right"
        elif right_blocked and not left_blocked:
            return "front_left"
        else:
            return "front"


# --- 4. Subroutines ---
def scan_for_aruco(stub, duration=0.5):
    """Rapidly polls the camera for ArUco markers and updates the live feed."""
    start_time = time.time()

    while time.time() - start_time < duration:
        try:
            response = stub.GetSensorData(rover_pb2.EmptyRequest())
            image_array = np.frombuffer(response.image, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if frame is not None:
                cv2.imshow("Rover Live Feed", frame)
                cv2.waitKey(1)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco_detector.detectMarkers(gray)

                if ids is not None and len(ids) > 0:
                    print(f"--> [ArUco Spotted! ID: {ids[0][0]}] ")
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.imshow("Rover Live Feed", frame)
                    cv2.waitKey(1)
                    return True
        except Exception:
            pass
    return False


def process_yolo(image_bytes: bytes):
    """Runs YOLO, updates live feed with bounding boxes, and calculates blocks."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"center": False, "left": False, "right": False}, []


    height, width, _ = frame.shape
    third_w = width // 3
    results = yolo_model.predict(frame, verbose=False)

    annotated_frame = results[0].plot()
    cv2.line(annotated_frame, (third_w, 0), (third_w, height), (0, 255, 255), 2)
    cv2.line(annotated_frame, (2 * third_w, 0), (2 * third_w, height), (0, 255, 255), 2)
    cv2.imshow("Rover Live Feed", annotated_frame)
    cv2.waitKey(1)

    cv_blocked = {"center": False, "left": False, "right": False}
    detected_obstacles = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area > MIN_OBJECT_AREA:
            cls_id = int(box.cls[0])
            detected_obstacles.append(yolo_model.names[cls_id])
            center_x = (x1 + x2) // 2

            if center_x < third_w:
                cv_blocked["left"] = True
            elif center_x > 2 * third_w:
                cv_blocked["right"] = True
            else:
                cv_blocked["center"] = True

    return cv_blocked, detected_obstacles


def delay_with_gui(seconds):
    """Replaces time.sleep() to ensure the video window doesn't freeze."""
    end_time = time.time() + seconds
    while time.time() < end_time:
        cv2.waitKey(50)

    # --- 5. Main Loop ---


def run_agent():
    # CHANGE THIS IP to the Raspberry Pi's IP when testing on physical hardware
    TARGET_IP = '10.79.69.64:50051'

    channel = grpc.insecure_channel(TARGET_IP)
    stub = rover_pb2_grpc.RoverControlStub(channel)  # Updated to new Service Name

    print(f"Connecting to Rover Server at {TARGET_IP}...")
    cv2.namedWindow("Rover Live Feed", cv2.WINDOW_NORMAL)

    while True:
        try:
            # Phase 1: ArUco Priority Scan
            print(f"\n[{time.strftime('%X')}] Scanning for ArUco...")
            aruco_detected = scan_for_aruco(stub, duration=0.5)

            # ArUco Interrupt Subroutine
            if aruco_detected:
                print("=" * 50)
                print(f"[{time.strftime('%X')}] 🚨 ARUCO INTERRUPT TRIGGERED 🚨")
                move_rover(stub, "stop")
                drop_payload(stub)
                move_rover(stub, "u_turn")

                print("[INTERRUPT] Executing U-Turn... waiting 3s.")
                delay_with_gui(3)

                print("[INTERRUPT] U-Turn Complete. Resuming navigation.")
                print("=" * 50)
                continue

                # Phase 2: Standard Navigation
            response = stub.GetSensorData(rover_pb2.EmptyRequest())
            us_data = list(response.ultrasonic)
            cv_blocked, detected_obstacles = process_yolo(response.image)

            print(f"--> Arduino US Data [L, F, R]: {us_data} | CV Blocked: {cv_blocked}")
            if detected_obstacles:
                print(f"--> YOLO Detected: {detected_obstacles}")

            # Fuse data, decide, and send command to Pi
            chosen_direction = decide_movement(us_data, cv_blocked)
            move_rover(stub, chosen_direction)

            delay_with_gui(TEST_DELAY_SECONDS)

        except grpc.RpcError as e:
            print(f"[ERROR] gRPC disconnected or Pi unreachable. Retrying...")
            delay_with_gui(2)
        except Exception as e:
            print(f"[ERROR] {e}")
            delay_with_gui(1)


if __name__ == '__main__':
    run_agent()