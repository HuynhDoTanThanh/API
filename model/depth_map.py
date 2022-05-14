import numpy as np
import torch
import cv2


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
        if value > 620:
            return 255
        if value < 345:
            return 0
        value = value/700*255

        value = (315*value - 39525)/(805-3*value)

        return int(value)

    def get_depth_map(self, image):
        input_batch = self.m_transform(image).to(self.device)

        
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
            depth_map = np.asarray(prediction)
            
        if self.obstacle:
            result = np.vectorize(self.transform_output)(depth_map)
            result = np.array(result, dtype='uint8')
        else:
            depth_map[depth_map>2000] = 2000
            depth_map[depth_map<1] = 1
            depth_map /= 2000
            result = depth_map
            
            # mean_depth_map = len(depth_map[depth_map>0.6])/(depth_map.shape[0]*depth_map.shape[1])
            # if mean_depth_map > 0.3:
            #     scale_func = lambda x: x + ((1/(x+1))-0.5)*(mean_depth_map-0.3)
            #     result = np.vectorize(scale_func)(depth_map)

        return result

    def get_obstacle(self, image, seg_lane_binary):
        depth_map = self.get_depth_map(image)

        seg_lane_binary = cv2.resize(np.array(seg_lane_binary, dtype='uint8'), self.size)

        obstacle = depth_map - seg_lane_binary*depth_map

        print(obstacle.shape)

        return np.array(obstacle, dtype='uint8')

    def check_obstacle(self, obstacle):
        obstacle_checker = []
        if np.sum(obstacle[:140, 200:350]) > 400000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[:140, 350:610]) > 400000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[:140, 610:760]) > 400000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[140:390, 200:350]) > 400000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[140:390, 350:610]) > 400000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[140:390, 610:760]) > 400000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[390:, 200:350]) > 400000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[390:, 350:610]) > 1000000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        if np.sum(obstacle[390:, 610:760]) > 400000:
            obstacle_checker.append(True)
        else:
            obstacle_checker.append(False)
        
        return obstacle_checker

    