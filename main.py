import configparser
import os
import time

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")


if __name__ == '__main__':
    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')
    username = os.environ['MQTT_USERNAME'] if 'MQTT_USERNAME' in os.environ else config['MQTT']['Username']
    password = os.environ['MQTT_PASSWORD'] if 'MQTT_PASSWORD' in os.environ else config['MQTT']['Password']
    host = os.environ['MQTT_HOST'] if 'MQTT_HOST' in os.environ else config['MQTT']['Host']
    port = int(os.environ['MQTT_PORT']) if 'MQTT_PORT' in os.environ else config['MQTT'].getint('Port')
    camera = config['YOLO'].getint('Camera')
    rapid_mode = config['YOLO'].getboolean('RapidMode')
    preview = config['YOLO'].getboolean('Preview')
    threshold = config['YOLO'].getfloat('Threshold')

    # Connect to MQTT broker
    print("Connecting to MQTT broker...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.username_pw_set(username, password)
    client.connect(host, port, 360)

    while not client.is_connected():
        client.loop()

    # Load camera
    print(f"Loading camera {camera}...")
    cap = cv2.VideoCapture(camera)
    print("Camera loaded")

    # Load YOLO
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded")

    # Main detection loop
    prev_people_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame, conf=threshold)
            if preview:
                cv2.imshow("YOLO to MQTT", results[0].plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            label_id = (list(results[0].names))[list(results[0].names.values()).index('person')]
            people_count = sum([1 for i in results[0].boxes.cls if i == label_id])
            print(f"Detected {people_count} results")

            if people_count != prev_people_count:
                client.publish("yolo/people_count", str(people_count))
                prev_people_count = people_count

            if prev_people_count > 0 and not rapid_mode:
                time.sleep(2)
        else:
            print("Failed to read frame, shutting down...")
            break

    # Cleanup
    client.disconnect()
    cap.release()
    cv2.destroyAllWindows()
