import cv2
import numpy as np
import torch
import pandas as pd
import imageio
from func import calc_dist_and_drawbox # import our fuction

# input video path
path = "pexels-adnan-karimi-7887761.mp4"
# path = 'pexels-rodnae-productions-8224214.mp4'

# Camera data
f = 25 # mm
sensor_h = 24 # mm
fps = 60
# fps = 30

# output
# output_path = 'output.avi' # empty for not save video
# output_path = 'output.gif'
output_slow_motion = True
stop_frame = 70 # N frames for output, 0 for all frames
show_frames = False # True for show frames due processing
speed_calc = True # calculate speed

# object data
real_size = 217 # bowl ball diameter in mm
# real_size = 67 # tennis ball size
target_label = 'sports ball'

# font and color
font = cv2.FONT_HERSHEY_SIMPLEX
color1 = (255,255,255)
color2 = (0,0,255)


#  load pretrained model yolov5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)

#  load video
cap = cv2.VideoCapture(path)



# Get the Default resolutions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



i = 1
h = 0
box_notfound_count = 0
frames = []

print(f'Start processing video {path}')
# main cycle
while True:

    # get start_frame
    success, img = cap.read()

    if not success: break

    # pass image trought net and get bounding boxes
    results = model([img])
    df = results.pandas().xyxy[0]

    # get boxes x,y,w,h for target_label and confidence above 0.5 and
    # calculate distance, speed, if no boxes add count
    box = df[(df.name == target_label) & (df.confidence >= 0.5)]
    if box.shape[0] > 0:
        box_notfound_count = 0
        box = box.iloc[0][0:4].apply(round).tolist()
        x, y, w, h, dist, speed = calc_dist_and_drawbox(img, box, i, fps,
                                        real_size, f, sensor_h,
                                        use_object_width = True,
                                        box_xywh = False,
                                        speed_calc = speed_calc
                                        )
    else:
        box_notfound_count +=1

    # draw distance and speed
    if (h>0) and (box_notfound_count < 3):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.rectangle(img, (x, y), (x+160, y-25), (0,0,255), -1)
        cv2.putText(img, f'Dist.:{dist:.2f}m', (x, y-10), font, 0.5, (255,255,255), 1)
        if speed_calc:
            cv2.rectangle(img, (x, y-25), (x+160, y-50), (0,0,255), -1)
            cv2.putText(img, f'Avg. speed:{speed:.2f}km/h', (x, y-30), font, 0.5, (255,255,255), 1)



    # mark frame if it in slowmotion
    if output_slow_motion:
        cv2.rectangle(img, (0, 0), (300, 15), (0,0,255), -1)
        cv2.putText(img, f'Slow motion enable (x 0.2)', (0, 10), font, 0.5, (255,255,255), 1)

    # add frame to our list
    frames.append(img)

    # show frame
    if show_frames:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow("Image", img)
        key = cv2.waitKey(60)

    if i == stop_frame: break
    i += 1

print(f'Finished processing. Made {len(frames)} frames')

# Release capture
cap.release()

frame_width = 360
frame_height = 640

# change fps if we need slowmotion
if output_slow_motion:
    fps = fps / 5

if len(output_path.split('.')) == 2:

    # write gif if output_path .gif
    if output_path.split('.')[1] == 'gif':
        with imageio.get_writer(output_path, mode="I", duration=1/fps) as gif_writer:
            for frame in frames:
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gif_writer.append_data(frame)

    # write avi in other cases
    else:
        out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
        for frame in frames:
            out.write(frame)
        out.release()

    print(f'Video write to {output_path} slow motion {output_slow_motion}')

# Release everything if job is finished
cv2.destroyAllWindows()
