from flask import Flask, Response, render_template, jsonify
import cv2
import math
import pyttsx3
import threading
import time
import numpy as np
import os

app = Flask(__name__)

# Load YOLOv2 model
yolo_dir = "yolo/"
weights_path = os.path.join(yolo_dir, "yolov3.weights")
config_path = os.path.join(yolo_dir, "yolov3.cfg")
names_path = os.path.join(yolo_dir, "coco.names")

# Check if the files exist
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"YOLO config file not found: {config_path}")
if not os.path.exists(names_path):
    raise FileNotFoundError(f"YOLO class names file not found: {names_path}")

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(names_path, "r") as f:
    classNames = [line.strip() for line in f.readlines()]

engine = pyttsx3.init()
detected_objects = set()
detected_objects_lock = threading.Lock()

# Global video capture object
cap = cv2.VideoCapture(0)

def release_camera():
    global cap
    if cap.isOpened():
        cap.release()

def gen_frames():
    global cap
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            current_objects = set()
            boxes = []
            confidences = []
            class_ids = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classNames[class_ids[i]])
                    current_objects.add(label)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
                    conf = confidences[i]
                    label_with_conf = f'{label} {conf:.2f}'
                    t_size = cv2.getTextSize(label_with_conf, 0, fontScale=1, thickness=2)[0]
                    c2 = x + t_size[0], y - t_size[1] - 3
                    cv2.rectangle(img, (x, y), c2, [255, 0, 255], -1, cv2.LINE_AA)
                    cv2.putText(img, label_with_conf, (x, y - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            
            with detected_objects_lock:
                detected_objects.update(current_objects)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def speak_detected_objects():
    last_spoken_objects = set()
    while True:
        with detected_objects_lock:
            if detected_objects:
                current_objects = set(detected_objects)
                new_objects = current_objects - last_spoken_objects
                if new_objects:
                    text = ', '.join(new_objects)
                    print(f"Speaking: {text}")  # Add logging
                    engine.say(text)
                    engine.runAndWait()
                    last_spoken_objects = current_objects
                else:
                    print("No new objects to speak.")
        time.sleep(3)  # Add a delay to prevent continuous speech

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detected_objects')
def get_detected_objects():
    global detected_objects
    with detected_objects_lock:
        return jsonify(list(detected_objects))

@app.route('/shutdown', methods=['POST'])
def shutdown():
    release_camera()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

if __name__ == '__main__':
    try:
        threading.Thread(target=speak_detected_objects, daemon=True).start()
        app.run(debug=True)
    except KeyboardInterrupt:
        release_camera()
        print('Camera released, application terminated.')
