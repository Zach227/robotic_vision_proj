import sys
import time
import argparse

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from coco_class_names import COCO_CLASS_LOOKUP
import numpy as np


import pyttsx3

WIDTH = 1920
HEIGHT = 1080


def draw_boxes(img, boxes):
    for i in range(boxes.xyxy.shape[0]):
        x1,y1,x2,y2 = boxes.xyxy[i]
        clas = boxes.cls[i]
        img = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 4)
        img = cv2.putText(img, COCO_CLASS_LOOKUP[int(clas)], (int(x1),int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img



def get_largest(boxes, item_class_num):
    largest_index = -1
    largest_area = -1

    for i in range(boxes.xyxy.shape[0]):
        x1,y1,x2,y2 = boxes.xyxy[i]
        area = abs(x1-x2) * abs(y1-y2)

        if item_class_num != None:
            if (area > largest_area) and boxes.cls[i] == item_class_num:
                largest_area = area
                largest_index = i
        else:
            if area > largest_area:
                largest_area = area
                largest_index = i

    return largest_index
        

        



parser = argparse.ArgumentParser()
# parser.add_argument("--mode", "-m", type=str, choices=["large", "item"], default="large", help="Path to model")
parser.add_argument("--item", "-i", type=str, default=None, help="item to look for")

args = parser.parse_args()

item_flag = False

if args.item != None:
    item_flag = True
    item_class_num = next((k for k, v in COCO_CLASS_LOOKUP.items() if v == args.item), None)
    if item_class_num == None:
        print("Item not found in class lookup")
        sys.exit(0)

model_path = "models/coco_128.pt"

model = YOLO(model_path)

picam2 = Picamera2()
# picam2.preview_configuration.main.size = (1024,768)
picam2.preview_configuration.main.size = (1920,1080)
# picam2.preview_configuration.main.size = (4608, 2592)

picam2.preview_configuration.main.format = "RGB888"
# picam2.preview_configuration.align()
# picam2.configure("preview")
picam2.start()

engine = pyttsx3.init()
last_location_string = ""

while True:
    try:
        frame = picam2.capture_array()
        results = model([frame])
        boxes = results[0].boxes
        if len(boxes.conf) == 0:
            print("No detection")
            continue


        # if item_flag:
        #     xy_vals = [ ]
        #     for i in range(len(boxes.cls)):
        #         if boxes.cls[i] == item_class_num:
        #             xy_vals.append(boxes.xyxy[i])
        # else:
        #     xy_vals = boxes.xyxy

        if item_flag:
            largest_index = get_largest(boxes, item_class_num)
            if largest_index == -1:
                print("Item not found")
                engine.say("Item not found")
                engine.runAndWait()
                time.sleep(.5)

                # img = draw_boxes(frame, boxes)
                # # filename = f"camera_images/{name}.png"
                # filename = f"camera_images/img.png"
                # cv2.imwrite(filename, img)
                # print(f"Saved {filename}")
                continue
        else:
            largest_index = get_largest(boxes, None)

        x1,y1,x2,y2 = boxes.xyxy[largest_index]
        center  = (int(abs(x1-x2)/2 + x1), int(abs(y1-y2)/2 + y1))
        print(f"Center: {center}")

        x_quad = ""
        y_quad = ""

        if center[0] < (WIDTH/3):
            x_quad = "left"
        elif center[0] < 2*(WIDTH/3):
            x_quad = "center"
        else:
            x_quad = "right"

        if center[1] < (HEIGHT/3):
            y_quad = "top"
        elif center[1] < 2*(HEIGHT/3):
            y_quad = "center"
        else:
            y_quad = "bottom"

        name = COCO_CLASS_LOOKUP[int(boxes.cls[largest_index])]

        location_string = name + " at " + x_quad + " " + y_quad
        print(location_string)

        if location_string != last_location_string:
            engine.say(location_string)
            engine.runAndWait()
            last_location_string = location_string
            time.sleep(.5)


                    
        # img = draw_boxes(frame, boxes)
        # if len(boxes.conf > 0):
        #     detect_string = COCO_CLASS_LOOKUP[int(boxes.cls[0])]
        # else:
        #     detect_string = "no detect"



        # img = draw_boxes(frame, boxes)
        # # filename = f"camera_images/{name}.png"
        # filename = f"camera_images/img.png"
        # cv2.imwrite(filename, img)
        # print(f"Saved {filename}")
        # time.sleep(2~)

    except KeyboardInterrupt:
        print("\nScript Terminated.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")



