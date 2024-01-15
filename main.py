import configparser
import time

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    slowdown = not config['YOLO'].getboolean('RapidMode')

    print("Connecting to MQTT broker...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.username_pw_set(config['MQTT']['Username'], config['MQTT']['Password'])
    client.connect(config['MQTT']['Host'], 1883, 360)

    while not client.is_connected():
        client.loop()

    camera = int(config['YOLO']['Camera']) if 'CAMERA' in config['YOLO'] else 0
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
            cv2.imshow("YOLO to MQTT", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            label_id = (list(results[0].names))[list(results[0].names.values()).index('person')]
            people_count = sum([1 for i in results[0].boxes.cls if i == label_id])
            print(f"Detected {people_count} results")

            if people_count != prev_people_count:
                client.publish("yolo/people_count", str(people_count))
                prev_people_count = people_count

            if prev_people_count > 0 and slowdown:
                time.sleep(2)
        else:
            print("Failed to read frame, shutting down...")
            break

    client.disconnect()
    cap.release()
    cv2.destroyAllWindows()
