import cv2
from retinaface import RetinaFace
from arcface_onnx import ArcFaceONNX
import numpy as np
from sklearn.preprocessing import Normalizer
import face_align
import faiss
import requests
import json


class FaceRecognition:
    def __init__(self, db_url, retinaface_model_path, arcface_model_type="openvino_128", embedding_dim=128):
        """
        Initialize the FaceRecognition system.

        Parameters
        ----------
        db_url : str
            URL to fetch face encoding data.
        retinaface_model_path : str
            Path to the RetinaFace ONNX model file.
        arcface_model_type : str, optional
            Model type for ArcFaceONNX, by default "openvino_128".
        embedding_dim : int, optional
            Dimension of face embeddings, by default 128.
        """
        self.db_url = db_url
        self.index_db = faiss.IndexFlatL2(embedding_dim)
        self.name_list, self.encodes = self.fetch_data()
        self.index_db.reset()
        self.index_db.add(self.encodes)

        self.detector = RetinaFace(model='openvino', model_file=retinaface_model_path)
        self.detector.prepare()

        self.recModel = ArcFaceONNX(model_type=arcface_model_type)
        self.l2_normalizer = Normalizer('l2')

    def fetch_data(self):
        """
        Fetch face encoding data from the database URL.

        Returns
        -------
        tuple
            person_image_ids (np.ndarray): Array of person IDs as strings.
            features_array (np.ndarray): Array of face feature embeddings.
        """
        response = requests.get(self.db_url)
        data = json.loads(response.text)
        features_array = np.array(
            [np.fromstring(item['feature'].strip("[]"), sep=" ") for item in data]
        )
        person_image_ids = np.array([item['id'] for item in data], dtype=str)
        return person_image_ids, features_array

    def process_image(self, images):
        """
        Extract and normalize embeddings from a list of face images.

        Parameters
        ----------
        images : list of np.ndarray
            List of cropped and aligned face images.

        Returns
        -------
        np.ndarray
            L2-normalized embeddings for the input images.
        """
        embeddings = np.vstack([self.recModel.get_feat(img) for img in images])
        embeddings = self.l2_normalizer.transform(embeddings)
        return embeddings

    @staticmethod
    def is_tiny(bbx, min_width=10, min_height=10):
        """
        Check if bounding boxes are tiny.

        Parameters
        ----------
        bbx : np.ndarray
            Bounding boxes array with shape (N, 5) [x1, y1, x2, y2, score].
        min_width : int, optional
            Minimum width to not be considered tiny, by default 10.
        min_height : int, optional
            Minimum height to not be considered tiny, by default 10.

        Returns
        -------
        tuple
            x1, y1, x2, y2 (np.ndarray): Adjusted bounding box coordinates.
            flag_tiny (np.ndarray): Boolean mask indicating tiny boxes.
        """
        x1, y1, x2, y2 = bbx[:, 0], bbx[:, 1], bbx[:, 2], bbx[:, 3]
        x1, y1, x2, y2 = np.maximum(x1, 0), np.maximum(y1, 0), np.maximum(x2, 0), np.maximum(y2, 0)
        w = x2 - x1
        h = y2 - y1
        flag_tiny = (w < min_width) | (h < min_height)
        return x1, y1, x2, y2, flag_tiny

    def calc_l2dis(self, embeddings, top_k=1):
        """
        Calculate L2 distances and retrieve top-k nearest neighbors.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to query.
        top_k : int, optional
            Number of nearest neighbors to retrieve, by default 1.

        Returns
        -------
        tuple
            distances (np.ndarray): Distances to nearest neighbors.
            indices (np.ndarray): Indices of nearest neighbors.
        """
        distances, indices = self.index_db.search(embeddings, top_k)
        return distances, indices

    def detect_and_recognize(self, img, confidence_threshold=0.6):
        """
        Detect faces in an image, extract embeddings, and find matches.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        confidence_threshold : float, optional
            Minimum confidence to consider a detection, by default 0.6.

        Returns
        -------
        tuple
            distances (np.ndarray): Distances to matched faces.
            indices (np.ndarray): Indices of matched faces.
            bboxes_img (list): List of bounding boxes for detected faces.
        """
        bboxes, kpss = self.detector.detect(img)
        bboxes, kpss = np.array(bboxes), np.array(kpss)
        print(f'fffffffffffffff:{bboxes}')

        if kpss is None or len(kpss) == 0:
            return None, None, []

        # Filter by confidence
        confidence_mask = bboxes[:, -1] > confidence_threshold
        bboxes_filtered = bboxes[confidence_mask]
        kpss_filtered = kpss[confidence_mask]

        # Filter out tiny faces
        _, _, _, _, flag_tiny = self.is_tiny(bboxes_filtered)
        non_tiny_mask = ~flag_tiny
        bboxes_filtered = bboxes_filtered[non_tiny_mask]
        kpss_filtered = kpss_filtered[non_tiny_mask]

        if len(bboxes_filtered) == 0:
            return None, None, []

        # Crop and align faces
        imgs = [face_align.norm_crop(img, landmark=kpts) for kpts in kpss_filtered]
        bboxes_img = bboxes_filtered[:, :4].tolist()

        # Extract embeddings and find matches
        embds = self.process_image(imgs)
        distances, indices = self.calc_l2dis(embds)

        return distances, indices, bboxes_img


if __name__ == "__main__":
    # Example usage
    IP = 'http://192.168.7.58:80'
    db_url = f'{IP}/api/face/encoding'
    retinaface_model_path = './vino/retinaface_mnet025_v2.onnx'

    face_recognizer = FaceRecognition(db_url, retinaface_model_path)

    img = cv2.imread('img_test.jpg')
    distances, indices, bboxes = face_recognizer.detect_and_recognize(img)

    if distances is not None:
        print("Distances:", distances)
        print("Indices:", indices)
        print("Bounding boxes:", bboxes)
    else:
        print("No faces detected or recognized.")
