from model.SemanticSegmentation.semantic_segmentation import models
from model.SemanticSegmentation.semantic_segmentation import load_model
from model.SemanticSegmentation.semantic_segmentation import draw_results

import torch
from torchvision import transforms
import cv2
import numpy as np

class SegLane(object):
    def __init__(self, weight_path="weights/model_BiSeNet-FullData-960_35_best.pt", threshold=0.4):
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

            results = results > self.threshold

        mask = results[0].cpu().numpy().astype("int")
        sidewalk_on_road = cv2.bitwise_or(mask[2], mask[1])
        crosswalk_on_sidewalk = cv2.bitwise_or(mask[2], mask[0])
        res = cv2.bitwise_or(mask[0], sidewalk_on_road)

        binary = res.copy()

        res += crosswalk_on_sidewalk + mask[0]
        res = np.array(res, dtype='uint8')
        return res, binary
    