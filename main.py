import os
import sys
import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("test/#")


def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.username_pw_set(os.environ['USER'], os.environ['PASS'])
client.connect(os.environ['HOST'], 1883, 60)

client.publish("test/sending", payload="Example", qos=0, retain=False)

client.loop_forever()
