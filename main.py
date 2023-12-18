import os

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")


if __name__ == '__main__':
    print("Connecting to MQTT broker...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.username_pw_set(os.environ['USER'], os.environ['PASS'])
    client.connect(os.environ['HOST'], 1883, 360)

    while not client.is_connected():
        client.loop()

    camera = int(os.environ['CAMERA']) if 'CAMERA' in os.environ else 0
    print(f"Loading camera {camera}...")
    cap = cv2.VideoCapture(camera)
    print("Camera loaded")

    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded")

    threshold = 0.25

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame, conf=threshold)
            cv2.imshow("YOLO to MQTT", results[0].plot())
            print(f"Detected {len(results[0])} results")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Failed to read frame, shutting down...")
            break

    client.disconnect()
    cap.release()
    cv2.destroyAllWindows()
