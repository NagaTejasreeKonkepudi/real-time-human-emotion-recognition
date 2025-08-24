import tkinter as tk
from tkinter import Canvas, Label, Text, END
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from threading import Thread
import os
import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

emotion_model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
try:
    emotion_model.load_weights('model.h5')
except Exception as e:
    print(f"Error loading model weights: {e}")

cv2.ocl.setUseOpenCL(False)

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Threaded webcam video stream class.
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            print("Error: Could not open webcam.")
            exit(1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.grabbed, self.frame = self.stream.read()
            if not self.grabbed:
                print("Warning: No frame grabbed from webcam.")
                break

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class VideoCamera:
    def __init__(self):
        self.cap1 = WebcamVideoStream(src=0).start()

    def get_frame(self):
        image = self.cap1.read()
        if image is None:
            return None, None

        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        error_levels = []
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            roi_gray_frame = gray[y:y+h, x:x+w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0
            )
            prediction = emotion_model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))
            confidence = int(prediction[0][maxindex] * 100)

            epsilon = 1e-10  # avoid log(0)
            p = prediction[0]
            entropy = -np.sum(p * np.log(p + epsilon))
            normalized_entropy = entropy / np.log(len(p))
            error_level = int(normalized_entropy * 100)
            error_levels.append(error_level)

            emotion_text = f"{emotion_dict[maxindex]}: {confidence}%"
            error_text = f"Error Level: {error_level}%"
            cv2.putText(image, emotion_text, (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, error_text, (x+20, y-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        display_error = max(error_levels) if error_levels else None
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), display_error

    def __del__(self):
        self.cap1.stop()

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.canvas = Canvas(window, width=600, height=500)
        self.canvas.pack()
        self.error_label = Label(window, text="Error Level: N/A", font=("Helvetica", 16), bg="grey", fg="white")
        self.error_label.pack(pady=10)
        self.error_log_text = Text(window, height=8, width=70)
        self.error_log_text.pack(pady=10)
        self.vid = VideoCamera()
        self.delay = 15  
        self.last_log_time = 0  
        self.last_error = None 
        self.update()
        self.window.mainloop()

    def update(self):
        frame, error_level = self.vid.get_frame()
        if frame is not None:
            img_arr = np.frombuffer(frame, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            self.photo = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        if error_level is not None:
            if error_level != 0:
                self.last_error = error_level
            elif self.last_error is not None:
                error_level = self.last_error

            self.error_label.config(text=f"Error Level: {error_level}%")
            if error_level > 50:
                self.error_label.config(bg="red", fg="white")
            else:
                self.error_label.config(bg="green", fg="white")
            
            current_time = time.time()
            if error_level > 50 and (current_time - self.last_log_time) > 2:
                log_text = f"{time.strftime('%H:%M:%S')} - High Error Level: {error_level}%\n"
                self.error_log_text.insert(END, log_text)
                self.error_log_text.see(END)
                self.last_log_time = current_time
        else:
            self.error_label.config(text="Error Level: N/A", bg="grey", fg="white")
        
        self.window.after(self.delay, self.update)

App(tk.Tk(), "Real-Time Emotion Detection with Persistent Error Level")
