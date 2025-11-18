# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

from __future__ import division
import numpy as np
import cv2
import onnx
import onnxruntime
# import face_align
import face_align
from openvino.runtime import Core
import time
import os

__all__ = ['ArcFaceONNX']

class ArcFaceONNX:
    def __init__(self, model_file=None, model_type='onnx', session=None, xml_file='./vino/w600k_mbf.xml', bin_file='./vino/w600k_mbf.bin'):
        """
        Initialize the model. Supports 'onnx', 'openvino' (512-dim), and 'openvino_128' (128-dim with batch optimization).

        Parameters:
        - model_type: 'onnx', 'openvino', or 'openvino_128'
        - model_file: Path to the ONNX model file (for 'onnx')
        - session: ONNX Runtime session (optional, for 'onnx')
        - xml_file: Path to the OpenVINO model's XML file (for 'openvino' or 'openvino_128')
        - bin_file: Path to the OpenVINO model's BIN file (for 'openvino' or 'openvino_128')
        """
        print(model_type)
        assert model_type in ['onnx', 'openvino', 'openvino_128']
        self.model_type = model_type


        self.taskname = 'recognition'
        self.ie = Core()

        if self.model_type == 'onnx':  
            assert model_file is not None
            self.model_file = model_file
            self.session = session or onnxruntime.InferenceSession(self.model_file, None)
            model = onnx.load(self.model_file)
            graph = model.graph
            find_sub = any(node.name.startswith(('Sub', '_minus')) for node in graph.node[:8])
            find_mul = any(node.name.startswith(('Mul', '_mul')) for node in graph.node[:8])
            self.input_mean = 0.0 if (find_sub and find_mul) else 127.5
            self.input_std = 1.0 if (find_sub and find_mul) else 127.5
            input_cfg = self.session.get_inputs()[0]
            self.input_shape = input_cfg.shape
            self.input_size = tuple(input_cfg.shape[2:4][::-1])
            self.input_name = input_cfg.name
            outputs = self.session.get_outputs()
            self.output_names = [out.name for out in outputs]
            assert len(self.output_names) == 1
            self.output_shape = outputs[0].shape

        elif self.model_type == 'openvino':  
            assert xml_file is not None and bin_file is not None
            self.net = self.ie.read_model(model=xml_file, weights=bin_file)
            self.compiled_model = self.ie.compile_model(self.net, "CPU")
            self.input_layer = (3, 112, 112)
            self.output_layer = next(iter(self.compiled_model.outputs))
            self.input_mean = 127.5
            self.input_std = 127.5
            self.input_shape = self.input_layer
            self.input_size = tuple(self.input_shape[1:4][::-1])

        elif self.model_type == 'openvino_128':  
            assert xml_file is not None and bin_file is not None
            self.model_path = "./vino/rec_mobile_128.xml"
            self.net = self.ie.read_model(model=self.model_path, weights=self.model_path.replace('xml', 'bin'))
            self.compiled_model = self.ie.compile_model(self.net, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
            self.input_layer = (3, 112, 112)
            self.output_layer = next(iter(self.compiled_model.outputs))
            self.input_mean = 0.5  
            self.input_std = 0.5   
            self.input_shape = (1, 3, 112, 112)  
            self.input_size = (112, 112)
            self.optimal_batch_size = None  

            self.scale = 1.0 / 255
            self.adjusted_mean =  [m / self.scale for m in [0.5, 0.5, 0.5]]
            self.blob_scalefactor = 1.0 / 127.5
            self.swapRB = False
            
    def prepare(self, ctx_id, **kwargs):
        if self.model_type == 'onnx' and ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        elif self.model_type == 'openvino_128':
            self.optimal_batch_size = self.find_optimal_batch_size()

    def run_openvino_inference(self, input_data, batch_size=None):
        """Runs inference with batch processing for openvino_128."""
        if batch_size is None:
            batch_size = self.optimal_batch_size or 1
        self.compiled_model([input_data[:batch_size]])  # Warm-up
        start_time = time.time()
        outputs = []
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i + batch_size]
            output = self.compiled_model([batch])[self.output_layer]
            outputs.append(output)
        return time.time() - start_time, np.concatenate(outputs, axis=0)


    

    def get(self, img, face):
        aimg = face_align.norm_crop(img, landmark=face)
        embedding = self.get_feat(aimg).flatten()
        return embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        
        if self.model_type == 'onnx':
            blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                          (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
            net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        elif self.model_type == 'openvino':
            blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                          (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
            net_out = self.compiled_model(blob)[self.output_layer]
        elif self.model_type == 'openvino_128':
            blob = cv2.dnn.blobFromImages(imgs, scalefactor=self.blob_scalefactor, size=input_size, mean=self.adjusted_mean, swapRB=self.swapRB)
            # scale = 1. / 255
            # adjusted_mean = [m / scale for m in [0.5, 0.5, 0.5]]
            # blob = cv2.dnn.blobFromImages(imgs, scalefactor=1.0 / 127.5 , size=input_size,
            #                               mean=adjusted_mean, swapRB=False)
            net_out = self.compiled_model(blob)[self.output_layer]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        if self.model_type == 'onnx':
            net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        elif self.model_type == 'openvino':
            net_out = self.compiled_model([blob])[self.output_layer]
        elif self.model_type == 'openvino_128':
            _, net_out = self.run_openvino_inference(blob, self.optimal_batch_size)
        return net_out




# # testing
# if __name__ == "__main__":

#     rec_128 = ArcFaceONNX(model_type='openvino_128', xml_file='rec_mobile_128.xml', bin_file='rec_mobile_128.bin')
#     rec_128.prepare(ctx_id=0)  
    
#     img = cv2.imread("/home/yazdan/programming/mehra/wide_camera/New-recog/g8lzmqp9hmljqvqewdgx.png")
#     if img is None:
#         print("Error: Could not load image.")
#         exit(1)

#     face_landmark = np.array([[50, 50], [70, 50], [60, 70], [55, 80], [65, 80]])  
    
#     embedding = rec_128.get(img, face_landmark)
#     print(f"Embedding shape: {embedding.shape}")
#     print(f"Sample embedding: {embedding[:]}")