import argparse
import datetime
import logging
import os
import random
import ssl
import time
import json

import jwt
import paho.mqtt.client as mqtt

import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

from ObjectDetectionHelper import ObjectDetection

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

global minimum_backoff_time
global MAXIMUM_BACKOFF_TIME

mqtt_bridge_port = 8883
mqtt_bridge_hostname = 'mqtt.googleapis.com'
cloud_region = 'us-central1'


device_id = "raspi400"
project_id = 'factory-automation-331600'
message_type = 'state'

registry_id = 'esteira'

num_messages = 100

private_key_file = '../.ssh/ec_private.pem'
ca_certs = "../.ssh/roots.pem"
algorithm = 'ES256'

# Publish to the events or state topic based on the flag.
sub_topic = "events" if message_type == "event" else "state"


mqtt_topic = '/devices/{}/{}'.format(device_id, sub_topic)


#Capturar sinal do sensor infravermelho

#Start IoT MQTT STUFF
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.CRITICAL)

# The initial backoff time after a disconnection occurs, in seconds.
minimum_backoff_time = 1

# The maximum backoff time before giving up, in seconds.
MAXIMUM_BACKOFF_TIME = 32

# Whether to wait with exponential backoff before publishing.
should_backoff = False


# [START iot_mqtt_jwt]
def create_jwt(project_id, private_key_file, algorithm):
    """Creates a JWT (https://jwt.io) to establish an MQTT connection.
    Args:
     project_id: The cloud project ID this device belongs to
     private_key_file: A path to a file containing either an RSA256 or
             ES256 private key.
     algorithm: The encryption algorithm to use. Either 'RS256' or 'ES256'
    Returns:
        A JWT generated from the given project_id and private key, which
        expires in 20 minutes. After 20 minutes, your client will be
        disconnected, and a new JWT will have to be generated.
    Raises:
        ValueError: If the private_key_file does not contain a known key.
    """

    token = {
        # The time that the token was issued at
        "iat": datetime.datetime.utcnow(),
        # The time the token expires.
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=20),
        # The audience field should always be set to the GCP project id.
        "aud": project_id,
    }

    # Read the private key file.
    with open(private_key_file, "r") as f:
        private_key = f.read()

    print(
        "Creating JWT using {} from private key file {}".format(
            algorithm, private_key_file
        )
    )

    return jwt.encode(token, private_key, algorithm=algorithm)
# [END iot_mqtt_jwt]


# [START iot_mqtt_config]
def error_str(rc):
    """Convert a Paho error to a human readable string."""
    return '{}: {}'.format(rc, mqtt.error_string(rc))


def on_connect(unused_client, unused_userdata, unused_flags, rc):
    """Callback for when a device connects."""
    print("on_connect", mqtt.connack_string(rc))

    # After a successful connect, reset backoff time and stop backing off.
    global should_backoff
    global minimum_backoff_time
    should_backoff = False
    minimum_backoff_time = 1


def on_disconnect(unused_client, unused_userdata, rc):
    """Paho callback for when a device disconnects."""
    print("on_disconnect", error_str(rc))

    # Since a disconnect occurred, the next loop iteration will wait with
    # exponential backoff.
    global should_backoff
    should_backoff = True


def on_publish(unused_client, unused_userdata, unused_mid):
    """Paho callback when a message is sent to the broker."""
    print("on_publish")


def on_message(unused_client, unused_userdata, message):
    """Callback when the device receives a message on a subscription."""
    payload = str(message.payload.decode("utf-8"))
    print(
        "Received message '{}' on topic '{}' with Qos {}".format(
            payload, message.topic, str(message.qos)
        )
    )


def get_client(
    project_id,
    cloud_region,
    registry_id,
    device_id,
    private_key_file,
    algorithm,
    ca_certs,
    mqtt_bridge_hostname,
    mqtt_bridge_port,
):
    """Create our MQTT client. The client_id is a unique string that identifies
    this device. For Google Cloud IoT Core, it must be in the format below."""
    client_id = "projects/{}/locations/{}/registries/{}/devices/{}".format(
        project_id, cloud_region, registry_id, device_id
    )
    print("Device client_id is '{}'".format(client_id))

    client = mqtt.Client(client_id=client_id)

    # With Google Cloud IoT Core, the username field is ignored, and the
    # password field is used to transmit a JWT to authorize the device.
    client.username_pw_set(
        username="unused", password=create_jwt(project_id, private_key_file, algorithm)
    )

    # Enable SSL/TLS support.
    client.tls_set(ca_certs=ca_certs, tls_version=ssl.PROTOCOL_TLSv1_2)

    # Register message callbacks. https://eclipse.org/paho/clients/python/docs/
    # describes additional callbacks that Paho supports. In this example, the
    # callbacks just print to standard out.
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Connect to the Google MQTT bridge.
    client.connect(mqtt_bridge_hostname, mqtt_bridge_port)

    # This is the topic that the device will receive configuration updates on.
    mqtt_config_topic = "/devices/{}/config".format(device_id)

    # Subscribe to the config topic.
    client.subscribe(mqtt_config_topic, qos=1)

    # The topic that the device will receive commands on.
    mqtt_command_topic = "/devices/{}/commands/#".format(device_id)

    # Subscribe to the commands topic, QoS 1 enables message acknowledgement.
    print("Subscribing to {}".format(mqtt_command_topic))
    client.subscribe(mqtt_command_topic, qos=0)

    return client


# [END iot_mqtt_config]


def detach_device(client, device_id):
    """Detach the device from the gateway."""
    # [START iot_detach_device]
    detach_topic = "/devices/{}/detach".format(device_id)
    print("Detaching: {}".format(detach_topic))
    client.publish(detach_topic, "{}", qos=1)
    # [END iot_detach_device]


def attach_device(client, device_id, auth):
    """Attach the device to the gateway."""
    # [START iot_attach_device]
    attach_topic = "/devices/{}/attach".format(device_id)
    attach_payload = '{{"authorization" : "{}"}}'.format(auth)
    client.publish(attach_topic, attach_payload, qos=1)
    # [END iot_attach_device]


def listen_for_messages(
    service_account_json,
    project_id,
    cloud_region,
    registry_id,
    device_id,
    gateway_id,
    num_messages,
    private_key_file,
    algorithm,
    ca_certs,
    mqtt_bridge_hostname,
    mqtt_bridge_port,
    jwt_expires_minutes,
    duration,
    cb=None,
):
    """Listens for messages sent to the gateway and bound devices."""
    # [START iot_listen_for_messages]
    global minimum_backoff_time

    jwt_iat = datetime.datetime.utcnow()
    jwt_exp_mins = jwt_expires_minutes
    # Use gateway to connect to server
    client = get_client(
        project_id,
        cloud_region,
        registry_id,
        gateway_id,
        private_key_file,
        algorithm,
        ca_certs,
        mqtt_bridge_hostname,
        mqtt_bridge_port,
    )

    attach_device(client, device_id, "")
    print("Waiting for device to attach.")
    time.sleep(5)

    # The topic devices receive configuration updates on.
    device_config_topic = "/devices/{}/config".format(device_id)
    client.subscribe(device_config_topic, qos=1)

    # The topic gateways receive configuration updates on.
    gateway_config_topic = "/devices/{}/config".format(gateway_id)
    client.subscribe(gateway_config_topic, qos=1)

    # The topic gateways receive error updates on. QoS must be 0.
    error_topic = "/devices/{}/errors".format(gateway_id)
    client.subscribe(error_topic, qos=0)

    # Wait for about a minute for config messages.
    for i in range(1, duration):
        client.loop()
        if cb is not None:
            cb(client)

        if should_backoff:
            # If backoff time is too large, give up.
            if minimum_backoff_time > MAXIMUM_BACKOFF_TIME:
                print("Exceeded maximum backoff time. Giving up.")
                break

            delay = minimum_backoff_time + random.randint(0, 1000) / 1000.0
            time.sleep(delay)
            minimum_backoff_time *= 2
            client.connect(mqtt_bridge_hostname, mqtt_bridge_port)

        seconds_since_issue = (datetime.datetime.utcnow() - jwt_iat).seconds
        if seconds_since_issue > 60 * jwt_exp_mins:
            print("Refreshing token after {}s".format(seconds_since_issue))
            jwt_iat = datetime.datetime.utcnow()
            client.loop()
            client.disconnect()
            client = get_client(
                project_id,
                cloud_region,
                registry_id,
                gateway_id,
                private_key_file,
                algorithm,
                ca_certs,
                mqtt_bridge_hostname,
                mqtt_bridge_port,
            )

        time.sleep(1)

    detach_device(client, device_id)

    print("Finished.")
    # [END iot_listen_for_messages]


def send_data_from_bound_device(
    service_account_json,
    project_id,
    cloud_region,
    registry_id,
    device_id,
    gateway_id,
    num_messages,
    private_key_file,
    algorithm,
    ca_certs,
    mqtt_bridge_hostname,
    mqtt_bridge_port,
    jwt_expires_minutes,
    payload,
):
    """Sends data from a gateway on behalf of a device that is bound to it."""
    # [START send_data_from_bound_device]
    global minimum_backoff_time

    # Publish device events and gateway state.
    device_topic = "/devices/{}/{}".format(device_id, "state")
    gateway_topic = "/devices/{}/{}".format(gateway_id, "state")

    jwt_iat = datetime.datetime.utcnow()
    jwt_exp_mins = jwt_expires_minutes
    # Use gateway to connect to server
    client = get_client(
        project_id,
        cloud_region,
        registry_id,
        gateway_id,
        private_key_file,
        algorithm,
        ca_certs,
        mqtt_bridge_hostname,
        mqtt_bridge_port,
    )

    attach_device(client, device_id, "")
    print("Waiting for device to attach.")
    time.sleep(5)

    # Publish state to gateway topic
    gateway_state = "Starting gateway at: {}".format(time.time())
    print(gateway_state)
    client.publish(gateway_topic, gateway_state)

    # Publish num_messages messages to the MQTT bridge
    for i in range(1, num_messages + 1):
        client.loop()

        if should_backoff:
            # If backoff time is too large, give up.
            if minimum_backoff_time > MAXIMUM_BACKOFF_TIME:
                print("Exceeded maximum backoff time. Giving up.")
                break

            delay = minimum_backoff_time + random.randint(0, 1000) / 1000.0
            time.sleep(delay)
            minimum_backoff_time *= 2
            client.connect(mqtt_bridge_hostname, mqtt_bridge_port)

        payload = "{}/{}-{}-payload-{}".format(registry_id, gateway_id, device_id, i)

        print(
            "Publishing message {}/{}: '{}' to {}".format(
                i, num_messages, payload, device_topic
            )
        )
        client.publish(device_topic, "{} : {}".format(device_id, payload))

        seconds_since_issue = (datetime.datetime.utcnow() - jwt_iat).seconds
        if seconds_since_issue > 60 * jwt_exp_mins:
            print("Refreshing token after {}s").format(seconds_since_issue)
            jwt_iat = datetime.datetime.utcnow()
            client = get_client(
                project_id,
                cloud_region,
                registry_id,
                gateway_id,
                private_key_file,
                algorithm,
                ca_certs,
                mqtt_bridge_hostname,
                mqtt_bridge_port,
            )

        time.sleep(5)

    detach_device(client, device_id)

    print("Finished.")
    # [END send_data_from_bound_device]

# [START iot_mqtt_run]


jwt_iat = datetime.datetime.utcnow()
jwt_exp_mins = 20
client = get_client(
    project_id, cloud_region, registry_id, device_id,
    private_key_file, algorithm, ca_certs,
    mqtt_bridge_hostname, mqtt_bridge_port)

objectDetection = ObjectDetection()
objectDetection.setUp()
objectDetection.startDetection()
#Start our infine loop
i = 0
while True:
    i = i + 1
    client.loop()

        # Wait if backoff is required.
    if should_backoff:
        # If backoff time is too large, give up.
        if minimum_backoff_time > MAXIMUM_BACKOFF_TIME:
            print("Exceeded maximum backoff time. Giving up.")
            break

        # Otherwise, wait and connect again.
        delay = minimum_backoff_time + random.randint(0, 1000) / 1000.0
        print("Waiting for {} before reconnecting.".format(delay))
        time.sleep(delay)
        minimum_backoff_time *= 2
        client.connect(mqtt_bridge_hostname, mqtt_bridge_port)

    # Publish "payload" to the MQTT topic. qos=1 means at least once
    # delivery. Cloud IoT Core also supports qos=0 for at most once
    # delivery.
    data_set = {"regId": registry_id, "devId": device_id, "num": i, "myData": random.randint(0, 100)}
    payload = json.dumps(data_set)
    
    print('Publishing message #{}: \'{}\''.format(i, payload))
        # [START iot_mqtt_jwt_refresh]
    seconds_since_issue = (datetime.datetime.utcnow() - jwt_iat).seconds
    if seconds_since_issue > 60 * jwt_exp_mins:
        print("Refreshing token after {}s".format(seconds_since_issue))
        jwt_iat = datetime.datetime.utcnow()
        client.loop()
        client.disconnect()
        client = get_client(
            project_id,
            cloud_region,
            registry_id,
            device_id,
            private_key_file,
            algorithm,
            ca_certs,
            mqtt_bridge_hostname,
            mqtt_bridge_port,
        )
    # [END iot_mqtt_jwt_refresh]
    # Publish "payload" to the MQTT topic. qos=1 means at least once
    # delivery. Cloud IoT Core also supports qos=0 for at most once
    # delivery.
    client.publish(mqtt_topic, payload, qos=1)

    # Send events every second. State should not be updated as often
    for i in range(0, 60):
        time.sleep(1)
        client.loop()
