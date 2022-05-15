from model.depth_map import DepthMap
from model.yolov5 import ObjectDetection
from model.bisenetv2 import SegLane
from util.draw import *
from util.depth2distance import depth_to_distance

import cv2
from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time


app = FastAPI()

Obstacle = DepthMap(model_type='MiDaS_small', size=(960,540), obstacle=True)
SegmentationLane = SegLane(weight_path="model/weights/model_BiSeNet-FullData-960_35_best.pt", threshold=0.4)
Object = ObjectDetection(weight_path='model/weights/best.pt', conf=0.5)
Distance = DepthMap(model_type='DPT_Hybrid', obstacle=False)

def read_imagefile(file) -> np.ndarray:
    buff = np.frombuffer(file, np.uint8)
    buff = buff.reshape(1, -1)
    image = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    return image   


@app.get('/index')
async def home():
  return "Hello World"

@app.post("/uploadfiles/")
async def create_upload_files(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    plt.imshow(image)
    plt.show()
    return "Read image successful"

@app.post("/streaming/")
async def streaming(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    dep_image = cv2.resize(image.copy(), (960, 540))
    seg_image = cv2.resize(image, (960, 960))

    start = time.time()
    lane, binary_lane = SegmentationLane.detect(seg_image)
    end1 = time.time()
    depth_obstacle = Obstacle.get_obstacle(dep_image, binary_lane)
    obstacle_checker = Obstacle.check_obstacle(depth_obstacle)
    end2 = time.time()
    print(end1 - start)
    print(end2 - end1)

    return obstacle_checker

@app.post("/distance/")
async def distance(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    image = cv2.resize(image, (960, 540))

    start = time.time()
    locate = Object.detect(image)

    depth_distance = Distance.get_depth_map(image)
    
    distances = []
    for _, center in locate:
        distances.append(depth_to_distance(depth_distance[center[1],center[0]]))
    end = time.time()
    print(end - start)

    return locate, distances





if __name__=="__main__":
    # image = cv2.imread("image/test_1.jpg")
    # image = cv2.resize(image, (960, 540))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Obstacle = DepthMap(model_type='MiDaS_small', size=(960,540), obstacle=True)
    # SegmentationLane = SegLane(weight_path="model/weights/model_BiSeNet-FullData-960_35_best.pt", threshold=0.4, size=(960, 960))
    # Object = ObjectDetection(weight_path='model/weights/best.pt', conf=0.5)
    # Distance = DepthMap(model_type='DPT_Hybrid', obstacle=False)

    # lane, binary_lane = SegmentationLane.detect(image)
    # depth_obstacle = Obstacle.get_obstacle(image, binary_lane)
    # obstacle_checker = Obstacle.check_obstacle(depth_obstacle)

    # locate = Object.detect(image)

    # depth_distance = Distance.get_depth_map(image)
    
    # distances = []
    # for _, center in locate:
    #     distances.append(depth_to_distance(depth_distance[center[1],center[0]]))

    # cv2.imshow('obstacle', draw_obstacle(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), obstacle_checker))
    # cv2.imshow('lane', cv2.resize(lane*70, (960, 540)))
    # cv2.imshow('object', draw_object_detecion(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), locate, distances))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    uvicorn.run(app, host="10.0.22.239", port=8000)