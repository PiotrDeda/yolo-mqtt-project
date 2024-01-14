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
    prev_people_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame, conf=threshold)
            people_count = sum([1 for label in results[0].names if label == 'person'])
            print(f"Detected {people_count} results")

            if people_count != prev_people_count:
                client.publish("people_count", str(people_count))
                prev_people_count = people_count

            cv2.imshow("YOLO to MQTT", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Failed to read frame, shutting down...")
            break

    client.disconnect()
    cap.release()
    cv2.destroyAllWindows()
