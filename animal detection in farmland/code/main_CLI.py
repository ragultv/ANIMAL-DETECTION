import cv2
import imutils
import numpy as np
import time
from collections import Counter
from threading import Thread
from imutils.video import FPS
import playsound
import os
from twilio.rest import Client

# Configuration for Twilio
FARM_OWNER_NUMBER = "+918838868452"  # Replace with the owner's number
TWILIO_ACCOUNT_SID = "ACbef0e7f9127fd39c1dc0894977f4e63e"  # Replace with your Twilio SID
TWILIO_AUTH_TOKEN = "7ab3a6d0d974b80a8331f75c241c8389"  # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "+14708237104"

# Paths
PROTO_PATH = "MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
SIREN_PATH = "Siren.wav"

# Load the pre-trained model
def load_model(proto_path, model_path):
    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Prototxt file not found: {proto_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Caffe model file not found: {model_path}")
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Send an SMS using Twilio
def send_sms(phone_number, detected_time):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Alert! Animal intrusion detected on your farm at {detected_time}.",
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        print("[INFO] SMS sent successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send SMS: {e}")
        return False

# Play the siren sound
def play_siren(siren_path):
    if not os.path.exists(siren_path):
        print(f"[ERROR] Siren file not found: {siren_path}")
        return
    try:
        playsound.playsound(siren_path, block=False)
    except Exception as e:
        print(f"[ERROR] Failed to play siren: {e}")

# Main animal detection function
def detect_animals():
    try:
        net = load_model(PROTO_PATH, MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "dining-table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "monitor"]
    REQ_CLASSES = {"bird", "cat", "cow", "dog", "horse", "sheep"}
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    print("[INFO] Starting video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)
    fps = FPS().start()

    conf_thresh = 0.2
    count = []
    flag = False

    while vs.isOpened():
        success, frame = vs.read()
        if not success:
            print("[ERROR] Video stream not accessible.")
            break

        frame = imutils.resize(frame, width=500)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        det = 0
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_thresh:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                if CLASSES[idx] in REQ_CLASSES:
                    det = 1
                    label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if (startY - 15) > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        cv2.imshow("Animal Detection", frame)
        count.append(det)

        if Counter(count[-36:])[1] > 15 and not flag:
            detected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[ALERT] Animal Intrusion Detected at {detected_time}!")
            play_siren(SIREN_PATH)
            send_sms(FARM_OWNER_NUMBER, detected_time)
            flag = True

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        fps.update()

    fps.stop()
    print(f"[INFO] Elapsed time: {fps.elapsed():.2f} seconds")
    print(f"[INFO] Approx. FPS: {fps.fps():.2f}")
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detection_thread = Thread(target=detect_animals, daemon=True)
    detection_thread.start()
    detection_thread.join()
