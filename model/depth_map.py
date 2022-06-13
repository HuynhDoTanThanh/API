from tracemalloc import start
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch
import cv2
import time


class DepthMap(object):
    def __init__(self, model_type='MiDaS_small', size=(960,540), obstacle=True):
        self.model_type = model_type
        self.size = size
        self.obstacle = obstacle

        self.get_model()

    def get_model(self):
        midas = torch.hub.load("intel-isl/MiDaS", self.model_type)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Device: ", device)
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        self.device = device
        self.midas = midas
        self.m_transform = transform
        return
    
    def transform_output(self, value):
        value = value/700

        value = (63*value-31)/(161-153*value)

        if value < 0:
            return 0
        elif value > 1:
            return 1
        else:
            return value

    def get_depth_map(self, image):
        input_batch = self.m_transform(image).to(self.device)

        start = time.time()
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy()

        end = time.time()

        time_predict = end - start

        if self.obstacle:
            result = np.vectorize(self.transform_output, otypes=[float])(depth_map)
        else:
            depth_map[depth_map>2500] = 2500
            depth_map[depth_map<1] = 1
            depth_map /= 2500
            result = depth_map

        return result, time_predict

    def get_obstacle(self, image, seg_lane_binary):
        depth_map, time_predict = self.get_depth_map(image)
        seg_lane_binary = cv2.resize(np.array(seg_lane_binary, dtype='uint8'), self.size)
        obstacle = depth_map - seg_lane_binary*depth_map

        return np.array(obstacle), time_predict

    def check_obstacle(self, obstacle):
        obstacle_checker = []
        if np.sum(obstacle[:140, 200:350]) > 1500:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[:140, 350:610]) > 1500:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[:140, 610:760]) > 1500:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[140:390, 200:350]) > 1500:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[140:390, 350:610]) > 1500:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[140:390, 610:760]) > 1500:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[390:, 200:350]) > 1500:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[390:, 350:610]) > 4000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[390:, 610:760]) > 1500:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        
        return obstacle_checker
