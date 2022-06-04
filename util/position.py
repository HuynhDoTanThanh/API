import numpy as np
import math

def get_position(points, mask):
    boundary = 50
    height = mask.shape[0]
    width = mask.shape[1]

    describe = []

    for p, c in points:
        start_point = [p[0] - int(boundary/2), p[1] - int(boundary/2)]
        end_point = [p[0] + int(boundary/2), p[1] + int(boundary/2)]

        if end_point[0] > height:
            end_point[0] = height
        if end_point[1] > width:
            end_point[1] = width
        if start_point[0] < 0:
            start_point[0] = 0
        if start_point[1] < 0:
            start_point[1] = 0
        
        crop_image = mask[start_point[0]:end_point[0], start_point[1]:end_point[1]]
            
        omega = np.sum(crop_image > 0)

        position = []

        #Sidewalk : 1, Road : 2 or Nothing : 0
        if omega < (boundary**2) / 4:
            position.append(0)
        else:
            avg = np.sum(crop_image) / omega 
            if avg > 1.5:
                position.append(1)
            else:
                position.append(2)

        #Left : 0, Center : 1 or Right : 2
        if p[0] < width * 0.3:
            position.append(0)
        elif p[0] < width * 0.7:
            position.append(1)
        else:
            position.append(2)
        
        #Far : 0 or Near : 1
        if p[1] < height * 0.5:
            position.append(0)
        else:
            position.append(1)
        
        position.append(direction(c))
        
        describe.append(position)

    return describe

def check_on_road(mask):
    crop_image = mask[900:, 430:530]
    omega = np.sum(crop_image > 0)
    if omega < 6000 / 4:
        return False
    else:
        avg = np.sum(crop_image) / omega
        if avg > 1.5:
            return False
        else:
            return True

def direction(point):
    angle_ob =  int(math.acos((480-point[0]) / math.sqrt((480-point[0])**2 + (960-point[1])**2))*180/math.pi)
    angle2oclock = [9,10,11,12,1,2,3]
    return angle2oclock[round(angle_ob/30)]


if __name__=='__main__':
    print(direction((200,200)))