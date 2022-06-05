from math import sqrt

last_box = (0,0,0,0)
last_dist = 0
distance = 0
start_frame = 0

# convert mm/s to km/h
def mmps_to_kmph(speed):
    return speed * 3600 / 10**6

# calculate distanse, speed and draw on image
def calc_dist_and_drawbox(img, box, frame_n, fps, real_size,
                        f = 25, # focus distance
                        sensor_h = 24, # camera sensor height mm
                        use_object_width = True, # use height for calc. dist.
                        box_xywh = True, # if false box x,y,x2,y2
                        threshold = 5, # threshold in px
                        speed_calc = True
                        ):

    global last_box, last_dist, distance, start_frame

    step_speed = 0
    dist_speed = 0

    # get frame size
    height, width, channels = img.shape

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

        step_path = sqrt(delta_dist**2 + delta_x**2 + delta_y**2)
        if step_path < (0.3 * real_size): step_path = 0

        distance += step_path

        if speed_calc:
            step_speed = step_path * fps
            if (start_frame == 0) and (distance > 0):
                start_frame = frame_n
                dist_speed = step_speed
            else:
                dist_speed = distance * fps / (frame_n - start_frame)

    last_box = box
    last_dist = dist

    return x, y, w, h, distance/10**3, mmps_to_kmph(dist_speed)
