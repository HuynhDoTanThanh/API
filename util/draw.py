import cv2
import numpy as np

def draw_object_detecion(image, result, distances):
    for (box, center), dis in zip(result, distances):
        cv2.rectangle(image, box[0], box[1], (0, 255, 0), 2)
        cv2.circle(image, center, 5, (0, 0, 255), 2)

        cv2.putText(image, str(np.round(dis,2)), (box[0][0], box[0][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)
    return image

def draw_obstacle(image, obstacle_checker):
    image = cv2.line(image, (200, 0), (200,560), (255, 0, 255), 1)
    image = cv2.line(image, (350, 0), (350,560), (255, 0, 255), 1)
    image = cv2.line(image, (610, 0), (610,560), (255, 0, 255), 1)
    image = cv2.line(image, (760, 0), (760,560), (255, 0, 255), 1)
    image = cv2.line(image, (200, 140), (760,140), (255, 0, 255), 1)
    image = cv2.line(image, (200, 390), (760,390), (255, 0, 255), 1)

    position = [(270,70),(480,70),(685,70),(270,265),(480,265),(685,265),(270,465),(480,465),(685,465)]
    for obs, p in zip(obstacle_checker, position):
        if obs:
            cv2.putText(image, "X", p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)

    return image

