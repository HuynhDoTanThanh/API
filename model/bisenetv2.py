from model.SemanticSegmentation.semantic_segmentation import models
from model.SemanticSegmentation.semantic_segmentation import load_model
from model.SemanticSegmentation.semantic_segmentation import draw_results
from util.position import get_position, check_on_road


import torch
from torchvision import transforms
import cv2
import numpy as np
class SegLane(object):
    def __init__(self, weight_path="weights/model_BiSeNet-960-2cat_46.pt", threshold=0.4):
        self.weight_path = weight_path
        self.threshold = threshold

        self.fn_image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.get_model()

    def get_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(self.weight_path, map_location=device)
        model = load_model(models["BiSeNetV2"], model)
        model.to(device).eval()

        self.device = device
        self.model = model
        return

    def detect(self, image):
        image = self.fn_image_transform(image)
        
        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0)
            results = self.model(image)['out']
            results = torch.sigmoid(results)
        
        result = results[0].cpu().numpy()
        result[result < 0.4] = 0 
        mask_road = result[0]
        mask_sidewalk = result[1]
        sidewalk_or_road = ((mask_road + mask_sidewalk) > 0.4).astype("int")
        binary = sidewalk_or_road.copy()
        get_max = (mask_sidewalk > mask_road).astype("int")
        res = sidewalk_or_road + get_max
        res = np.array(res, dtype='uint8')
        on_road = check_on_road(res)
        return res, binary, on_road
        
    def describe(self, image, points):
        image = self.fn_image_transform(image)

        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0)
            results = self.model(image)['out']
            results = torch.sigmoid(results)
            
        result = results[0].cpu().numpy()
        result[result < 0.4] = 0 
        mask_road = result[0]
        mask_sidewalk = result[1]
        sidewalk_or_road = ((mask_road + mask_sidewalk) > 0.4).astype("int")
        get_max = (mask_sidewalk > mask_road).astype("int")
        res = sidewalk_or_road + get_max
        res = np.array(res, dtype='uint8')

        return get_position(points, res)