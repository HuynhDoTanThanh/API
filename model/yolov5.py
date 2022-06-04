import torch
import time
import numpy as np

class ObjectDetection(object):
    def __init__(self, weight_path='weights/best.pt', conf=0.5):
        self.weight_path = weight_path
        self.conf = conf
        self.get_model()

    def get_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weight_path)
        model.conf = self.conf
        self.model = model

    def detect(self, image):
        start = time.time()
        pre = self.model(image, size=640)
        end = time.time()
        time_predict = end - start

        locate = pre.pandas().xyxy[0]

        result = []

        for _, row in locate.iterrows():
            result.append([
                            [int((row['xmax'] + row['xmin']) / 2), int(row['ymax'])],
                            [int((row['xmax'] + row['xmin']) / 2), int((row['ymax'] + row['ymin']) / 2)],
                            row['name'], 
                            [(int(row['xmin']), int(row['ymin'])),(int(row['xmax']), int(row['ymax']))]
                        ])
         
        return np.array(result, dtype='object'), time_predict
