import cv2
import numpy as np
from math import sqrt
import torch
import pandas as pd

# convert mm/s to km/h
def mmps_to_kmph(speed):
    return speed * 3600 / 10**6

# calculate distanse, speed and draw on image.
# return bounding box x, y, width, height, path distance(mm), speed(km/h)
def calc_dist_and_drawbox(img, box, frame_n, fps, real_size,
                        f = 25, # focus distance
                        use_object_width = True, # use height for calc. dist.
                        box_xywh = True, # if false box x,y,x2,y2
                        threshold = 5 # threshold in px
                        ):

    global last_box, last_dist, distance, start_frame

    step_speed = 0
    dist_speed = 0

    if not box_xywh:
        x, y, x2, y2 = box
        w = abs(x2-x)
        h = abs(y2-y)
    else:
        x, y, w, h = box

    x_last, y_last, w_last, h_last = last_box

    # calc. distance and it changes
    if use_object_width: dist = (f * real_size * width) / (w * sensor_h)
    else: dist = (f * real_size * height) / (h * sensor_h)
    delta_dist = dist - last_dist

    if last_dist > 0:

        # x/y movemens in px
        delta_x = abs(x - x_last)
        delta_y = abs(y - y_last)

        # controll threshold
        if delta_x < threshold: delta_x = 0
        if delta_y < threshold: delta_y = 0
        if abs(h-h_last) < threshold: delta_dist = 0

        # convert x/y movemens in mm
        delta_x = delta_x * real_size / w
        delta_y = delta_y * real_size / h

        # calc. distanse at step
        step_path = sqrt(delta_dist**2 + delta_x**2 + delta_y**2)

        # ignore dist. < 30% of object size
        if step_path < (0.3 * real_size): step_path = 0

        # calc total path dist. and speed at step
        distance += step_path
        step_speed = step_path * fps

        #  cal avg. speed
        if (start_frame == 0) and (distance > 0):
            start_frame = frame_n
            dist_speed = step_speed
        else:
            dist_speed = distance * fps / (frame_n - start_frame)

    last_box = box
    last_dist = dist

    return x, y, w, h, distance/10**3, mmps_to_kmph(dist_speed)






# Camera data
f = 25 # mm
sensor_h = 24 # mm
fps = 60

# object data
# class_id = 32 # sport ball class id
real_size = 217 # disc diameter
target_label = 'sports ball'

# font and color
font = cv2.FONT_HERSHEY_SIMPLEX
color1 = (255,255,255)
color2 = (0,0,255)

#  load video
cap = cv2.VideoCapture("pexels-rodnae-productions-8224214.mp4")

#  load pretrained model yolov5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)

last_box = (0,0,0,0)
last_dist = 0
distance = 0
start_frame = 0
i = 1
h = 0
box_notfound_count = 0

# main cycle
while True:

    # get start_frame
    success, img = cap.read()
    if not success: continue

    # get frame size
    height, width, channels = img.shape


    # pass image trought net and get bounding boxes
    results = model([img])
    df = results.pandas().xyxy[0]

    # get boxes x,y,w,h for target_label and confidence above 0.5 and
    # calculate distance, speed, if no boxes add count
    box = df[(df.name == target_label) & (df.confidence >= 0.5)]
    if box.shape[0] > 0:
        box_notfound_count = 0
        box = box.iloc[0][0:4].apply(round).tolist()
        x, y, w, h, dist, speed = calc_dist_and_drawbox(img, box, i, fps, real_size, use_object_width = True, box_xywh = False)
    else:
        box_notfound_count +=1

    # draw distance and speed
    if (h>0) and (box_notfound_count < 3):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.rectangle(img, (x, y), (x+160, y-50), (0,0,255), -1)
        cv2.putText(img, f'Dist.:{dist:.2f}m', (x, y-10), font, 0.5, (255,255,255), 1)
        cv2.putText(img, f'Speed:{speed:.2f}km/h', (x, y-30), font, 0.5, (255,255,255), 1)
    i += 1

    # show image
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow("Image", img)
    key = cv2.waitKey(90)
    # key = cv2.waitKey(0)



cv2.destroyAllWindows()
