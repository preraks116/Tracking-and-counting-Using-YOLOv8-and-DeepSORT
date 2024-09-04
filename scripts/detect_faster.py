# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect_faster.py --source <vid-path>.mp4
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from csv import writer
from paddleocr import PaddleOCR,draw_ocr

#basically abhi jo kiya wahi karna hai saare splitted videos ke saath. 
#Tu pura loop bhi chala sakta hai chahe to, matlab woh points ka location hamesha same hi hai actually, aur naming bhi maine order me hi kiya hai... lekin abi loop lagaya nahi hu....


# import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import pyshine as ps # for drawing

countClickingTimes = 0
rect2Points = []

def get_time_value_from_ocr_framePart(rect2Points, im0):
    x1 = rect2Points[0][0]
    y1 = rect2Points[0][1]
    x2 = rect2Points[1][0]
    y2 = rect2Points[1][1]
    crop_time = im0[y1:y2, x1:x2]
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
    # cv2.imwrite("./tmp.png", crop_time)
    # img_path = './tmp.png'
    # result = ocr.ocr(img_path, cls=True)
    result = ocr.ocr(crop_time, cls=True)
    result = result[0]
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    txts = txts[0]
    # scores = [line[1][1] for line in result]
    cv2.rectangle(im0, rect2Points[0], rect2Points[1], (0,0,255), 2)
    cv2.putText(im0, str(txts), (x1, y1-10),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
    return im0, txts

def draw_rectangle(frame1):
    def click_and_mark_points_event(event, x, y, flags, params): # function to display the coordinates of the points clicked on the image
        global rect2Points, rectangleCount, countClickingTimes
        
        if event == cv2.EVENT_LBUTTONDOWN: # checking for left mouse clicks
            # print(x, ' ', y) # displaying the coordinates on the Shell
            imgFrameToDrawOn = cv2.circle(frame1, (x,y), 5, (0, 255, 255), cv2.FILLED)
            cv2.imshow('Mark Points', imgFrameToDrawOn)  # Displaying the image
            
            if(countClickingTimes == 2):
                countClickingTimes = countClickingTimes + 1
                rect2Points.append((x,y))
                # rect2PointsDict: [(841, 434), (957, 515), (834, 594), (480, 491)]
            else:
                countClickingTimes = countClickingTimes + 1
                rect2Points.append((x,y))


    while 1:
        cv2.namedWindow('Mark Points', cv2.WINDOW_NORMAL)
        cv2.imshow('Mark Points', frame1) # displaying the image
        cv2.setMouseCallback('Mark Points', click_and_mark_points_event) # setting mouse handler for the image and calling the click_event() function
        cv2.waitKey(0)
        cv2.destroyWindow('Mark Points')
        break
    
    print("_____________________________________________________"+str(rect2Points))

    return rect2Points

def get_iou_peron_and_motorcycle(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of person (sitting on motorcycle) and motorcycle.
    If IOU is sufficient, only then I will consider that person as someone sitting on bike and only then we will distinguish that person from a pedestrian
    Or, if a person's IOU is not overlapping with motorcycle that means he his a pedestrian. 
    Parameters
    ----------
    bb1 : dict of motorcycle
        Keys: ['x1', 'x2', 'y1', 'y2']
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict of person
        Keys: ['x1', 'x2', 'y1', 'y2']
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    
    assert bb1[0] <= bb1[1]
    assert bb1[2] <= bb1[3]
    assert bb2[0] <= bb2[1]
    assert bb2[2] <= bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0] + 1) * (bb1[3] - bb1[2] + 1)
    bb2_area = (bb2[1] - bb2[0] + 1) * (bb2[3] - bb2[2] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_circle_BB(whole_frame):
    x_circle = 0
    y_circle = 0
    r = 0

    # Convert to grayscale.
    gray = cv2.cvtColor(whole_frame, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                param2 = 30, minRadius = 18, maxRadius = 19)

    # Draw circles that are detected.
    if detected_circles is not None:
        # print("detected_circles[0]", detected_circles[0])
        # print("detected_circles ", detected_circles)
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        # print("detected_circles[0, :] ", detected_circles[0, :])
        
        first_circle = detected_circles[0, :][0]
        # for pt in detected_circles[0, :]:
            # print("pt ", pt)
            # x_circle, y_circle, r = pt[0], pt[1], pt[2]
        x_circle, y_circle, r = first_circle[0], first_circle[1], first_circle[2]
        print(f"\n\n\nradius: {r} \n\n\n ")

    return x_circle, y_circle, r

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5x.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=[0, 2, 3, 5, 7],  # class 0, 2, 3 and 5 for person, car, motorcycle, bus, truck
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    ctr = 0
    old_clk_value = ''
    new_clk_value = ''
    skip = False
    file_name = str(source) + '.csv'
    print(file_name)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # print("classes: ", model.names)
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            tv_count = 0 #tv=total vehicle
            tp_count = 0 #tp=total person
            motorcycle_count = 0
            person_on_motorcycle = 0
            motorcycle_coord = []
            person_coord = []
            bicycle_coord = []
            car_coord = []
            bus_coord = []
            truck_coord = []
            all_coord = []
            target_circle_on_person_on_motorcycle = []
            target_circle_on_pedestrian_or_motorcycleperson = []
            target_circle_on_pedestrian = []
            target_circle = ''
            flag = False 
            
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                frame1 = im0.copy()
            
            if ctr == 0:
                rect2Points = draw_rectangle(frame1)
                #rect2Points = [(104, 844), (193, 867)]
                print("_____________________________________________rect2Points: "+ str(rect2Points))
            
            im0, click_time = get_time_value_from_ocr_framePart(rect2Points, im0)
            

            ''' Do all the processing and save in excel only on every new click '''
            old_clk_value = click_time
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if (len(det) and (new_clk_value != old_clk_value)):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    if(c.item()==2 or c.item()==1 or c.item()==3 or c.item()==5 or c.item()==7): # car, bicycle, motorcycle, bus, truck
                        tv_count = tv_count + n.item()
                    if(c.item()==3): #motorcycle
                        motorcycle_count = n.item()
                    if(c.item()==0): #person
                        tp_count = n.item()

                # get coordinates of tracker (circle)
                x_circle, y_circle, r = get_circle_BB(im0)
                # print("x_circle ", x_circle)
                # print("y_circle ", y_circle)
                # print("radius", r)
                # Draw the circumference of the circle.
                if(x_circle != 0 or y_circle != 0):
                    cv2.circle(im0, (x_circle, y_circle), r, (0, 255, 0), 2)
                    # Draw a small circle (of radius 1) to show the center.
                    cv2.circle(im0, (x_circle, y_circle), 1, (0, 0, 255), 3)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # object coordinates
                    (x, y) = (int(xyxy[0]), int(xyxy[1]))
                    (w, h) = ((int(xyxy[2]) - int(xyxy[0])), (int(xyxy[3])-int(xyxy[1])))

                    if(cls.item()==3):
                        motorcycle_coord.append([x,x+w,y,y+h])
                    
                    if(cls.item()==0):
                        person_coord.append([x,x+w,y,y+h])

                    if(cls.item()==1):
                        bicycle_coord.append([x,x+w,y,y+h])

                    if(cls.item()==2):
                        car_coord.append([x,x+w,y,y+h])

                    if(cls.item()==5):
                        bus_coord.append([x,x+w,y,y+h])

                    if(cls.item()==7):
                        truck_coord.append([x,x+w,y,y+h])

                    
                    if(cls.item()==0 or cls.item()==2 or cls.item()==1 or cls.item()==3 or cls.item()==5 or cls.item()==7): # person, car, bicycle, motorcycle, bus, truck
                        all_coord.append([x,x+w,y,y+h])

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)   

                # print("person count before cls.item(): ", tp_count)
                # print("motorcycle count before: ", motorcycle_count)
                if motorcycle_count > 0 and tp_count > 0: # if motorcycle_count = 0, means people detected are all pedestrians, if person_count = 0 means, bikes detected are all parked (not to be counted in density)
                    for each_person in person_coord:
                        for each_motorcycle in motorcycle_coord:
                            iou = get_iou_peron_and_motorcycle(each_motorcycle, each_person)
                            # print("IOU ", iou)
                            if (iou >= 0.05): # 8%
                                person_on_motorcycle = person_on_motorcycle + 1
                                # print("person_on_motorcycle: ", person_on_motorcycle)
                                # print("x,y,w,h: ", each_person)
                                target_circle_on_person_on_motorcycle.append(each_person)
                            
                            elif (iou < 0.05):
                                # is circle touching/inside this pedestrian
                                target_circle_on_pedestrian_or_motorcycleperson.append(each_person)

                    tp_count = tp_count - person_on_motorcycle
                    
                    if(tp_count < 0):
                        tp_count = 0
                
                for each_bicycle in bicycle_coord:
                    if(each_bicycle[0]<=x_circle<=each_bicycle[1] and each_bicycle[2]<=y_circle<=each_bicycle[3]):
                        target_circle = "vehicle"

                for each_car in car_coord:
                    if(each_car[0]<=x_circle<=each_car[1] and each_car[2]<=y_circle<=each_car[3]):
                        target_circle = "vehicle"
                
                for each_bus in bus_coord:
                    if(each_bus[0]<=x_circle<=each_bus[1] and each_bus[2]<=y_circle<=each_bus[3]):
                        target_circle = "vehicle"

                for each_truck in truck_coord:
                    if(each_truck[0]<=x_circle<=each_truck[1] and each_truck[2]<=y_circle<=each_truck[3]):
                        target_circle = "vehicle"

                for each_motorcycle in motorcycle_coord:
                    if(each_motorcycle[0]<=x_circle<=each_motorcycle[1] and each_motorcycle[2]<=y_circle<=each_motorcycle[3]):
                        target_circle = "vehicle"

                for each_person in target_circle_on_pedestrian_or_motorcycleperson:
                    if each_person not in target_circle_on_person_on_motorcycle:
                        target_circle_on_pedestrian.append(each_person)
                # for each_person in target_circle_on_person_on_motorcycle:
                #     if each_person not in target_circle_on_pedestrian_or_motorcycleperson:
                #         target_circle_on_pedestrian.append(each_person)

                for each_person in target_circle_on_pedestrian:
                    # print("pedestrian: ", each_person)
                    if(each_person[0]<=x_circle<=each_person[1] and each_person[2]<=y_circle<=each_person[3]):
                        target_circle = "person"

                # print("target_circle_on_person_on_motorcycle: ", target_circle_on_person_on_motorcycle)
                for each_person in target_circle_on_person_on_motorcycle:
                    # is circle touching/inside this motorcycle or person on motorcycle
                    # print("person on motorcycle: ", each_person)
                    if(each_person[0]<=x_circle<=each_person[1] and each_person[2]<=y_circle<=each_person[3]):
                        # print("target circle on this person on motorcycle")
                        target_circle = "vehicle"


                for each_bb in all_coord:
                    if(each_bb[0]<=x_circle<=each_bb[1] and each_bb[2]<=y_circle<=each_bb[3]):
                        flag = True
                
                if flag==False:
                    target_circle = "on/off road"

                if(x_circle==0 and y_circle==0):
                    target_circle = "___"
                
                # print("tp_count: ", tp_count)
                # print("motorcycle_count: ", motorcycle_count)
                # print("***************************************************")
                im0 =  ps.putBText(im0, "ctr " + "TV " + "  TP  " + "Target ", text_offset_x=10,text_offset_y=10,vspace=10,hspace=10, font_scale=1.0,background_RGB=(255,0,0),text_RGB=(255,250,250))
                im0 =  ps.putBText(im0, str(ctr).zfill(3) + " " + str(tv_count).zfill(3) + " " + str(tp_count).zfill(3) + " " + target_circle, text_offset_x=10,text_offset_y=60,vspace=10,hspace=10, font_scale=1.0,background_RGB=(255,0,0),text_RGB=(255,250,250))
                # print("TV === ", tv_count)
                # print("TP === ", tp_count)
                # print("Target === ", target_circle)
                
                # List that we want to add as a new row
                List = [ctr, click_time, tv_count, tp_count, target_circle]
                # old_clk_value = click_time
                # Open our existing CSV file in append mode
                # Create a file object for this file
                
                with open(file_name, 'a') as f_object:
                    # Pass this file object to csv.writer()
                    # and get a writer object
                    writer_object = writer(f_object)
                    # Pass the list as an argument into
                    # the writerow()
                    if new_clk_value != old_clk_value and skip == False:
                        writer_object.writerow(List)
                    # Close the file object
                    f_object.close()
                
            new_clk_value = old_clk_value
            # Stream results
            im0 = annotator.result()

            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        ctr = ctr + 1

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5x.pt', help='model path or triton URL')
    # parser.add_argument('--source', type=str, default='./Recording055_Trim.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source', type=str, default='../all_vids/Tobii Pro Glasses Analyzer 2022-11-07 23-50-18.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--source', type=str, default='/home/deepti/research/Dynamic_Saliency_Analysis_15Oct/4Nov22_vehicleDensity/trackingVids_ffmpeg/orig_vids/aw730QtQBmzxN-RJP9ON3g==OUT.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--source', type=str, default='/home/deepti/research/Dynamic_Saliency_Analysis_15Oct/4Nov22_vehicleDensity/trackingVids_ffmpeg/orig_vids/omjyf5FN3OaYsLoI48_02w==OUT.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--source', type=str, default='/home/deepti/research/Dynamic_Saliency_Analysis_15Oct/4Nov22_vehicleDensity/trackingVids_ffmpeg/orig_vids/R19lHCbYp8g7u3gCmy5_2g==OUT.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--source', type=str, default='/home/deepti/research/Dynamic_Saliency_Analysis_15Oct/4Nov22_vehicleDensity/trackingVids_ffmpeg/orig_vids/us90a3Vfj6wscqlwSe3Vcg==OUT.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.47, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default='False', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


# def main(opt):
#     check_requirements(exclude=('tensorboard', 'thop'))
#     run(**vars(opt))
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    
    # Assuming 'source' in 'opt' is now a list of video paths
    for video_path in opt.source:
        # Update the 'source' attribute in 'opt' with the current video path
        opt.source = video_path
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    video_paths = [
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-39.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-40.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-41.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-42.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-43.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-44.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-45.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-46.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-47.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-48.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-49.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-50.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-51.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-52.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-53.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-54.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-55.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-56.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-57.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-58.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-59.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-60.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-61.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-62.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-63.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-64.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-65.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-66.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-67.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-1.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-2.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-3.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-4.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-5.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-6.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-7.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-8.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-9.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-10.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-11.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-12.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-13.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-14.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-15.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-16.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\rec17-17.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-18.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-19.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-20.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-21.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-22.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-23.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-24.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-25.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-26.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-27.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-28.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-29.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-30.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-31.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-32.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-33.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-34.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-35.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-36.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-37.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-38.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-87.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-88.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-89.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-90.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-91.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-92.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-93.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-94.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-99.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-100.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-101.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-102.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-103.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-104.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-105.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-106.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-107.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-108.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-109.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-39.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-40.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-41.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-42.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-43.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-44.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-45.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-46.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-47.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-48.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-49.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-50.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-51.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-52.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-53.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-54.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-55.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-56.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-57.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-58.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-59.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-60.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-61.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-62.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-63.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-64.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-65.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-66.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-67.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-68.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-69.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-70.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-71.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-72.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-73.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-74.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-75.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-76.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-77.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-78.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-79.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-80.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-81.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-82.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-83.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-84.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-85.mp4",
"D:\\Two_Wheeler_vids\\trimmed\\Rec17\\Rec17-86.mp4"
]
    opt.source = video_paths
    main(opt)

