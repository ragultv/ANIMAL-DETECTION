import tkinter as tk
from tkinter import filedialog
from threading import Thread
import numpy as np
import cv2
import imutils
import time
from collections import Counter
import playsound
from cv2 import VideoCapture
from cv2.dnn import Net
from imutils.video import FPS

# Initialize Tkinter
app = tk.Tk()
app.title("Animal Detection App")

# GUI Components
video_label = tk.Label(app, text="Select Video:")
video_label.pack(pady=10)

selected_video = tk.StringVar()


def browse_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    selected_video.set(file_path)


browse_button = tk.Button(app, text="Browse", command=browse_video)
browse_button.pack(pady=10)


class Siren:
    pass


def start_detection(siren=None):
    proto = ("MobileNetSSD_deploy.prototxt.txt")
    model = ("MobileNetSSD_deploy.caffemodel")

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "dining-table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train",
               "monitor"]
    REQ_CLASSES = ["bird", "cat", "cow", "dog", "horse", "sheep"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    net: Net = cv2.dnn.readNetFromCaffe(proto, model)

    vs: VideoCapture = cv2.VideoCapture(selected_video.get(), cv2.CAP_FFMPEG)
    time.sleep(2)
    fps = FPS().start()

    conf_thresh = 0.2
    count = []
    flag = 0
    c = 0  # Initialize c
    siren_wav = "Siren.wav"

    while vs:
        success, frame = vs.read()
        if not success:
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
                    label = "{}: {:.2f}%".format("Animal", confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (36, 255, 12), 2)
                    if (startY - 15) > 15:
                        y = (startY - 15)
                    else:
                        y = (startY + 15)
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        cv2.imshow("Frame", frame)
        count.append(det)

        if flag == 1 and len(count) > c + (11 * 18):
            flag = 0
        if Counter(count[len(count) - 36:])[1] > 15 and flag == 0:
            print(f"Animal Intrusion Alert...!!! {len(count)}")
            playsound.playsound(siren_wav, block=False)
            flag = 1
            c = len(count)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        fps.update()

    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approximate FPS: {:.2f}".format(fps.fps()))
    vs.release()
    cv2.destroyAllWindows()


start_button = tk.Button(app, text="Start Detection", command=lambda: Thread(target=start_detection).start())
start_button.pack(pady=10)

# Run the Tkinter event loop
app.mainloop()
