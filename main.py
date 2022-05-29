import base64
from matplotlib import image
from model.classification_question import ClassificationQuestion
from model.depth_map import DepthMap
from model.yolov5 import ObjectDetection
from model.bisenetv2 import SegLane
from model.speech_to_text import Speech2Text
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
import json
import argparse


app = FastAPI()

Obstacle = DepthMap(model_type='MiDaS_small', size=(960,540), obstacle=True)
SegmentationLane = SegLane(weight_path="model/weights/model_BiSeNet-960-2cat_46.pt", threshold=0.4)
Object = ObjectDetection(weight_path='model/weights/best.pt', conf=0.5)
Distance = DepthMap(model_type='DPT_Hybrid', obstacle=False)
Question = ClassificationQuestion(model_path="model/weights/classification_question.pkl")
SpeechRecognition = Speech2Text()

def read_imagefile(file) -> np.ndarray:
    img_base64_string = file["data"].replace("\n", "")

    img_bytes = base64.b64decode(img_base64_string)
    img_bytesIO = BytesIO(img_bytes)
    img_bytesIO.seek(0)
    image = Image.open(img_bytesIO)
    return np.array(image) 


@app.get('/index')
async def home():
  return "Hello World"

@app.post('/name')
async def name(n: str):
  return "Hello World " + n

@app.post("/uploadfiles")
async def create_upload_files(file):
    image = read_imagefile(file)
    print(image.shape)
    return "Read image successful"

@app.post("/streaming")
async def streaming(file: dict):
    image = read_imagefile(file)

    dep_image = cv2.resize(image.copy(), (960, 540))
    seg_image = cv2.resize(image, (960, 960))

    start = time.time()
    _, binary_lane, on_road = SegmentationLane.detect(seg_image)
    end1 = time.time()
    depth_obstacle = Obstacle.get_obstacle(dep_image, binary_lane)
    obstacle_checker = Obstacle.check_obstacle(depth_obstacle)
    end2 = time.time()
    print(end1 - start)
    print(end2 - end1)

    output = {'obstacle': obstacle_checker, 'on_road': on_road}
    
    return json.dumps(output)

@app.post("/describe")
async def describe(file: dict):
    image = read_imagefile(file)
    seg_image = cv2.resize(image.copy(), (960, 960))
    image = cv2.resize(image, (960, 540))
    question = SpeechRecognition.recognition(file["ques"])
    print(question)
    if question == None:
        return "0"
    else:
        class_ques = ["Road", "Sidewalk", "Left", "Right", "Front", "All", "Near", "Far"]
        start = time.time()
        focus_region = Question.predict(question)[0]
        # print(destination)
        locate = Object.detect(image)
        if len(locate) == 0:
            return "-1"
        else:
            depth_distance = Distance.get_depth_map(image)

            positions = np.array(SegmentationLane.describe(seg_image, locate[:,0:2]))
            '''
            Value 1: {Sidewalk : 1, Road : 2, Nothing : 0}
            Value 2: {Left : 0, Front : 1, Right : 2}
            Value 3: {Far : 0, Near : 1}
            '''
            object_names = locate[:,2]
            get_distance = lambda x: depth_to_distance(depth_distance[x[1], x[0]])
            distances= np.vectorize(get_distance)(locate[:,1])

            # print(positions.shape, object_names.shape, distances.shape)
            result = np.concatenate((positions, object_names.reshape(-1,1), distances.reshape(-1,1)), axis=1)
            
            # print(result)

            if focus_region == 0:
                result = result[result[:, 0] == 2]
            elif focus_region == 1:
                result = result[result[:, 0] == 1]
            elif focus_region == 2:
                result = result[result[:, 1] == 0]
            elif focus_region == 3:
                result = result[result[:, 1] == 2]
            elif focus_region == 4:
                result = result[result[:, 1] == 1]
            elif focus_region == 5:
                pass
            elif focus_region == 6:
                result = result[result[:, 2] == 1]
            else:
                result = result[result[:, 2] == 0]
            
            # print(len(result))
            result = result[result[:, -1].argsort()][:, 3:]

            result_dict = {"orientation": result[:, 0].tolist(), "object_name": result[:, 1].tolist(), "distance": result[:, 2].tolist()}



            # print(result)
            end = time.time()
            print(end - start)

            output = {'result': result_dict, 'focus_region': class_ques[focus_region]}

            return json.dumps(output)





if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process create API')
    parser.add_argument('-H', '--host', type=str,
                        help='host of API')
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=8000)