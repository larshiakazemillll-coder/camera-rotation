import threading
import queue
import time
import cv2
import numpy as np
from ultralytics import YOLO
from retinaface import RetinaFace
from arcface_onnx import ArcFaceONNX
from sklearn.preprocessing import Normalizer
import face_align
import faiss
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.auth import HTTPDigestAuth



class FaceRecognition:
    def __init__(self, db_url, retinaface_model_path, arcface_model_type="openvino_128", embedding_dim=128):
        self.db_url = db_url
        self.embedding_dim = embedding_dim
        self.index_db = faiss.IndexFlatL2(embedding_dim)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        
        self.name_list, self.encodes = self.fetch_data()
        if self.encodes.size > 0 and len(self.encodes.shape) == 2:
            self.index_db.add(self.encodes)
        else:
            print("Warning: No face encodings available. Face recognition will be skipped.")

        self.detector = RetinaFace(model='openvino', model_file=retinaface_model_path)
        self.detector.prepare()

        self.recModel = ArcFaceONNX(model_type=arcface_model_type)
        self.l2_normalizer = Normalizer('l2')

    def fetch_data(self):
        try:
            response = self.session.get(self.db_url, timeout=30, stream=True)
            response.raise_for_status()
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
            data = json.loads(content.decode('utf-8'))
            if not data:
                print(f"Database at {self.db_url} is empty.")
                return np.array([]), np.array([])
            features_array = np.array(
                [np.fromstring(item['feature'].strip("[]"), sep=" ") for item in data]
            )
            person_image_ids = np.array([item['id'] for item in data], dtype=str)
            return person_image_ids, features_array
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {self.db_url}: {e}")
            return np.array([]), np.array([])

    def process_image(self, images):
        embeddings = np.vstack([self.recModel.get_feat(img) for img in images])
        embeddings = self.l2_normalizer.transform(embeddings)
        return embeddings

    @staticmethod
    def is_tiny(bbx, min_width=10, min_height=10):
        x1, y1, x2, y2 = bbx[:, 0], bbx[:, 1], bbx[:, 2], bbx[:, 3]
        x1, y1, x2, y2 = np.maximum(x1, 0), np.maximum(y1, 0), np.maximum(x2, 0), np.maximum(y2, 0)
        w = x2 - x1
        h = y2 - y1
        flag_tiny = (w < min_width) | (h < min_height)
        return x1, y1, x2, y2, flag_tiny

    def calc_l2dis(self, embeddings, top_k=1):
        if self.encodes.size == 0 or len(self.encodes.shape) != 2:
            print("No face encodings in database. Skipping recognition.")
            return None, None
        distances, indices = self.index_db.search(embeddings, top_k)
        return distances, indices

    def detect_and_recognize(self, img, confidence_threshold=0.6):
        bboxes, kpss = self.detector.detect(img)
        bboxes, kpss = np.array(bboxes), np.array(kpss)

        if kpss is None or len(kpss) == 0:
            return None, None, []

        confidence_mask = bboxes[:, -1] > confidence_threshold
        bboxes_filtered = bboxes[confidence_mask]
        kpss_filtered = kpss[confidence_mask]

        _, _, _, _, flag_tiny = self.is_tiny(bboxes_filtered)
        non_tiny_mask = ~flag_tiny
        bboxes_filtered = bboxes_filtered[non_tiny_mask]
        kpss_filtered = kpss_filtered[non_tiny_mask]

        if len(bboxes_filtered) == 0:
            return None, None, []

        imgs = [face_align.norm_crop(img, landmark=kpts) for kpts in kpss_filtered]
        bboxes_img = bboxes_filtered[:, :4].tolist()

        embds = self.process_image(imgs)
        distances, indices = self.calc_l2dis(embds)

        return distances, indices, bboxes_img


class DahuaPTZController:
    def __init__(self, ip, username, password, channel=1):
        self.ip = ip
        self.channel = channel
        self.auth = HTTPDigestAuth(username, password)

    def move_directly(self, arg1, arg2, arg3):
        url = (f"http://{self.ip}/cgi-bin/ptz.cgi?action=moveDirectly"
               f"&channel={self.channel}&code=screen"
               f"&arg1={arg1:.3f}&arg2={arg2:.3f}&arg3={arg3:.3f}")
        try:
            response = requests.get(url, auth=self.auth, timeout=2)
            response.raise_for_status()
            print(f"PTZ command sent successfully: {url}")
        except Exception as e:
            print(f"Failed to send PTZ command: {e}")

    def reset_position(self):
        url = (f"http://{self.ip}/cgi-bin/ptz.cgi?action=start"
               f"&channel={self.channel}&code=PositionABS"
               f"&arg1=250&arg2=16&arg3=0")
        try:
            response = requests.get(url, auth=self.auth, timeout=2)
            response.raise_for_status()
            print(f"PTZ reset command sent successfully: {url}")
        except Exception as e:
            print(f"Failed to send PTZ reset command: {e}")

    def move_right(self, speed=50, duration=1):
        url = (f"http://{self.ip}/cgi-bin/ptz.cgi?action=start"
               f"&channel={self.channel}&code=Right"
               f"&arg1={speed}&arg2={duration}&arg3=0")
        try:
            response = requests.get(url, auth=self.auth, timeout=2)
            response.raise_for_status()
            print(f"PTZ command sent successfully: {url}")
        except Exception as e:
            print(f"Failed to send PTZ command: {e}")

    def move_left(self, speed=50, duration=1):
        url = (f"http://{self.ip}/cgi-bin/ptz.cgi?action=start"
               f"&channel={self.channel}&code=Left"
               f"&arg1={speed}&arg2={duration}&arg3=0")
        try:
            response = requests.get(url, auth=self.auth, timeout=2)
            response.raise_for_status()
            print(f"PTZ command sent successfully: {url}")
        except Exception as e:
            print(f"Failed to send PTZ command: {e}")

    def stop_movement(self):
        url = (f"http://{self.ip}/cgi-bin/ptz.cgi?action=stop"
               f"&channel={self.channel}&code=Right"
               f"&arg1=50&arg2=1&arg3=20")
        try:
            response = requests.get(url, auth=self.auth, timeout=2)
            response.raise_for_status()
            print(f"PTZ stop command sent successfully: {url}")
            time.sleep(1.7)
        except Exception as e:
            print(f"Failed to send PTZ stop command: {e}")

def detect_person_yolov8(frame, model, conf_threshold=0.5):
    results = model(frame, conf=conf_threshold, classes=[0])  # class 0 = person
    boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            boxes.append([x1, y1, x2, y2])
    return boxes

def read_frame_with_retry(cap, retries=5, delay=0.5):
    for _ in range(retries):
        ret, frame = cap.read()
        if ret:
            return ret, frame
        time.sleep(delay)
    return False, None

def frame_reader_thread(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if ret:
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass
        time.sleep(1/30)  # Approximate 30 fps

def detection_thread(frame_queue, control_queue, yolo_model):
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            persons = detect_person_yolov8(frame, yolo_model, conf_threshold=0.5)
            if len(persons) > 0:
                # Send a simple 'detect' command; PTZ thread will handle coordinates
                control_queue.put(('detect',))
            frame_queue.task_done()
        except queue.Empty:
            continue

def ptz_control_thread(ptz, control_queue, cap, face_recognizer, yolo_model):
    while True:
        for direction in ['right', 'left']:
            if direction == 'right':
                ptz.move_right(speed=50, duration=1)
            else:
                ptz.move_left(speed=50, duration=1)
            
            start_time = time.time()
            while time.time() - start_time < 3:
                try:
                    command = control_queue.get_nowait()
                    if command[0] == 'detect':
                        ptz.stop_movement()
                        # Capture a new frame after stopping
                        ret, frame = read_frame_with_retry(cap)
                        if ret:
                            # Re-run YOLO on the stationary frame
                            persons = detect_person_yolov8(frame, yolo_model, conf_threshold=0.5)
                            if len(persons) > 0:
                                # Use the first detected person
                                x1, y1, x2, y2 = persons[0]
                                # Calculate zoom coordinates
                                frame_h, frame_w = frame.shape[:2]
                                x1 = max(0, min(x1, frame_w))
                                y1 = max(0, min(y1, frame_h))
                                x2 = max(0, min(x2, frame_w))
                                y2 = max(0, min(y2, frame_h))
                                height = y2 - y1
                                y2_new = y1 + height / 3
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2_new) / 2
                                arg2 = max(0, min(y_center / frame_h, 1))
                                arg1 = max(0, min(x_center / frame_w, 1))
                                face_area = (x2 - x1) * (y2_new - y1)
                                arg3 = max(0, min(face_area / (frame_h * frame_w), 0.5))
                                # Zoom to the new coordinates
                                ptz.move_directly(arg1, arg2, arg3)
                                time.sleep(5)
                                ret2, zoomed_frame = read_frame_with_retry(cap)
                                if ret2:
                                    cv2.imwrite("zommed frame.jpg", zoomed_frame)
                                    distances, indices, bboxes = face_recognizer.detect_and_recognize(zoomed_frame)
                                    if distances is not None:
                                        print("Recognition distances:", distances)
                                        print("Recognition indices:", indices)
                                        print("Bounding boxes:", bboxes)
                                ptz.reset_position()
                                time.sleep(3)
                                # Clear the queue
                                while not control_queue.empty():
                                    try:
                                        control_queue.get_nowait()
                                    except queue.Empty:
                                        break
                                break  # Resume rotation cycle
                            else:
                                print("Person not found in stationary frame; resuming rotation")
                                break
                except queue.Empty:
                    pass
                time.sleep(0.1)  # Check every 0.1 seconds
            else:
                ptz.stop_movement()  # Stop after movement if no detection

def main():
    dahua_ip = "192.168.7.108"
    ptz_username = "admin"
    ptz_password = "Aa123456!"
    db_ip = "192.168.7.40:80"

    ptz = DahuaPTZController(ip=dahua_ip, username=ptz_username, password=ptz_password)
    yolo_model = YOLO("yolov8n.pt")
    face_recognizer = FaceRecognition(
        db_url=f'http://{db_ip}/api/face/encoding',
        retinaface_model_path='./vino/retinaface_mnet025_v2.onnx'
    )

    cap = cv2.VideoCapture('rtsp://admin:Aa123456!@192.168.7.108:554/cam/realmonitor?channel=1&subtype=0')

    frame_queue = queue.Queue(maxsize=10)
    control_queue = queue.Queue()

    frame_reader = threading.Thread(
        target=frame_reader_thread,
        args=(cap, frame_queue),
        daemon=True
    )
    detection = threading.Thread(
        target=detection_thread,
        args=(frame_queue, control_queue, yolo_model),
        daemon=True
    )
    ptz_control = threading.Thread(
        target=ptz_control_thread,
        args=(ptz, control_queue, cap, face_recognizer, yolo_model),
        daemon=True
    )

    frame_reader.start()
    detection.start()
    ptz_control.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down threads")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()