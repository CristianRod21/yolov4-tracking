from __future__ import print_function
import sys
import cv2
from random import randint
import time
import tensorflow as tf
from imutils import perspective
from imutils import contours
from core.config import cfg

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import imutils

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/road.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', False, 'disable cv2 window during the process') # this is good for the .ipynb
flags.DEFINE_string('tracker_type', 'CSRT', 'the tracking algorithm [BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE,CSRT] ')

def createTrackerByName(tracker_type):# Set up tracker.
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker

def get_detections(frame, input_size, infer):
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    prev_time = time.time()

    if FLAGS.framework == 'tflite':
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
    else:
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
        
    return  boxes, scores, classes, valid_detections

def update_tracked_objects(frame, input_size, boxes,scores,classes,valid_detections):
    tmp_frame = cv2.resize(frame, (input_size, input_size))
    image_h, image_w, _ = input_size,input_size,input_size

    # Initialize Tracking
    tracker_type = FLAGS.tracker_type
    
    trackers_dict = dict()
    
    newBboxes = []
    
    for bbox in boxes[0]:
            bbox = tuple(bbox.tolist()) 
            coor = [None] * 4
            coor[0] = int(bbox[0] * image_h) # y1
            coor[2] = int(bbox[2] * image_h) # y2
            coor[1] = int(bbox[1] * image_w) # x1
            coor[3] = int(bbox[3] * image_w) # x2

            # Covertion for multitracker
            x1 = int(coor[1])
            y1 = int(coor[0])
            width = int(coor[2]) - int(coor[0]) # y2-y1 
            height = int(coor[3])- int(coor[1]) # x2-x1

            #print(f'x1 {x1}  y1 {y1}  heigh: {height} width {width} ')

            coords = tuple([x1,y1,height,width]) 

            newBboxes.append(coords)
            

    trackers_dict = dict()    
    for box in newBboxes:
        if (box[0] > 0 and box[1] > 0 and box[2] > 0 and box[3] > 0):
            trackers_dict[box] = createTrackerByName(tracker_type)
            trackers_dict[box].init(frame, box)
            
    return newBboxes, trackers_dict, tmp_frame

def order_points_old(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def update_tracker(trackers_dict,frame, newBboxes,scores,classes  ):
    del_items = []
    print('My dream is to fly')
    for idx, items in enumerate(trackers_dict.items()):
        
        obj,tracker = items

        #print(obj, tracker)
        ok, bbox = None, None

        try:
            ok, bbox = tracker.update(frame)
            if (ok):
                # Updates bbox
                newBboxes[idx] = bbox
            else:
                print('Failed to track ', obj)
                del_items.append(obj)
        
        except:
            print('Error')
            del_items.append(obj)

    # Deletes if failed to track
    for idx, item in enumerate(del_items):
        bounding_box_index = newBboxes.index(item)
        print(bounding_box_index, item)
        # Deletes object from tracker           
        trackers_dict.pop(item)
        #print(f'Before deleting {len(newBboxes)}')
        # Deletes object from bboxes
        newBboxes.pop(bounding_box_index)
        #print(f'After deleting {len(newBboxes)}')
        # Deletes object from scores
        scores =  np.expand_dims(np.delete(scores[0], bounding_box_index), axis=0)
        # Deletes object from classes
        classes = np.expand_dims(np.delete(classes[0], bounding_box_index), axis=0)
    
    return newBboxes,scores,classes 
        

def get_iou_2(box1, box2, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    
    
    ##height = y2-y1
    #width = x2-x1
    #x1,y1,height,width
    
    # Desired fomat is x1,y1,x2,y2
    a = (box1[0], box1[1], box1[0]+box1[3], box1[1]+box1[2])
    b = (box2[0], box2[1], box2[0]+box2[3], box2[1]+box2[2])

    
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_id = 0
    
    
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
    else:
        if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
            print("Video processing complete")
        raise ValueError("No image! Try with another video format")
    
    # Get initial bounding boxes at time_step 0
    boxes, scores, classes, valid_detections  = None,None,None,None
    newBboxes, tmp_frame = None,None
    trackers_dict = dict()
        
    # Frames
    prev_frame_time = 0
    new_frame_time = 0
    
    # Contains the most recent bounding boxes of the tracker
    oldBoxes = []
    newBboxes = []
    # For every image frame
    while True:
        # Reads the image
        return_value, frame = vid.read()
        
        new_frame_time = time.time() 
        original_h, original_w, original_c = frame.shape
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        
        # Variable declarations
        boxes,scores,classes,valid_detections,newBboxes
        
        allowed_classes = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'train']
        
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        
        # Resizes to yolo size
        new_frame = cv2.resize(frame, (input_size, input_size))

        ######## If its the first frame, or every 10th frames update the detector##########
        if (frame_id % 1 == 0 ):
            print(f"Frame {frame_id}, updating detections")
            
            # Updates bboxes from YOLO
            boxes, scores, classes, valid_detections  = get_detections(frame, input_size, infer)
            
            
            # Delete detections that are not in allowed_classes
            deleted_indx = []
            
            for i in range(len(boxes[0])): 
                class_indx = int(classes[0][i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
            
            bboxes = np.expand_dims(np.delete(boxes[0], deleted_indx, axis=0), axis=0)
            scores = np.expand_dims(np.delete(scores[0], deleted_indx, axis=0), axis=0)
            classes = np.expand_dims(np.delete(classes[0], deleted_indx, axis=0), axis=0)
            
            tmp_frame = cv2.resize(frame, (input_size, input_size))
            image_h, image_w, _ = input_size,input_size,input_size

            # Initialize Tracking
            tracker_type = FLAGS.tracker_type
            # Contains all YOLO predictions
            newBboxes = []
            
            # Loop over YOLO bboxes and change from y1,y2,x1,x2 format to x1,y1,width,height
            for bbox in bboxes[0]:
                    bbox = tuple(bbox.tolist()) 
                    coor = [None] * 4
                    coor[0] = int(bbox[0] * image_h) # y1
                    coor[2] = int(bbox[2] * image_h) # y2
                    coor[1] = int(bbox[1] * image_w) # x1
                    coor[3] = int(bbox[3] * image_w) # x2

                    # Covertion for multitracker
                    x1 = int(coor[1])
                    y1 = int(coor[0])
                    width = int(coor[2]) - int(coor[0]) # y2-y1 
                    height = int(coor[3])- int(coor[1]) # x2-x1

                    #print(f'x1 {x1}  y1 {y1}  heigh: {height} width {width} ')

                    coords = tuple([x1,y1,height,width]) 
                    if(coords[0] > 0 and coords[1] > 0 and coords[2] > 0 and coords[3] > 0):
                        newBboxes.append(coords)
            

            # Index of the new bounding box that overlaps with an existing bbox
            overlapping_boxes_indexes = []
            
            
            IOU_THRES = 0.5
            # Iterates through every new bounding box
            for i, detectorBox in enumerate(newBboxes):
                # Max overlapp that occurs within i newBbox and j oldBox
                max_iou_box_index = 0
                max_iou_box_score = 0
                trackerBoxFix = None
                # Iterates through every old (tracker) bounding box
                for j,trackerBox in enumerate(oldBoxes):
                    # Calculates the intersection over union
                    iou = get_iou_2(detectorBox, trackerBox ) 
                    print(f'YOLO box {i} overlapped with existing Tracker box {j} with an IOU of: {iou}')
                    #input('Kys to continue...')
                    if (iou > IOU_THRES):
                        # If the new IOU is greater than one found before, updates
                        if (iou > max_iou_box_score ):
                            max_iou_box_index = j
                            max_iou_box_score = iou
                            trackerBoxFix = trackerBox
                if (max_iou_box_score > IOU_THRES):
                    # Overlapping with existing bbox DONT add it to tracker
                    overlapping_boxes_indexes.append(i)
                    
            for i, box in enumerate(newBboxes):
                if (box[0] > 0 and box[1] > 0 and box[2] > 0 and box[3] > 0):
                    if (i not in overlapping_boxes_indexes):
                        trackers_dict[box] = createTrackerByName(tracker_type)
                        trackers_dict[box].init(frame, box)
                        oldBoxes.append(box)
                    else:
                        print(f'Skipping box {i} existing overlapp')
            # Updates the objects to be tracked
            #newBboxes, trackers_dict, tmp_frame = update_tracked_objects(new_frame, input_size, bboxes,scores,classes,valid_detections)
            input('STOP IT')
            
        #print(scores.shape, classes.shape)
        #print(oldBoxes)
        # Updates the tracked objects
        # IMPORTANT
        tracker_boxes,scores,classes  = update_tracker(trackers_dict,new_frame, oldBoxes ,scores,classes )

        input(f'Old boxes {oldBoxes}')
        
        # Pack and print the bbox
        pred_bbox = [tracker_boxes, scores, classes, valid_detections.numpy()]
        image = utils.draw_bbox_tracker(new_frame, pred_bbox)
        

        # TODO: Apply non-maxima supression?
        
        # Prints FPS
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time 
        fps = str(int(fps)) 
        cv2.putText(result, fps, (70, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA)
        
       
        # Changin old boxes
        oldBoxes = tracker_boxes
        input(f'{len(oldBoxes)}')

        if not FLAGS.dis_cv2_window:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if FLAGS.output:
            out.write(result)
        frame_id += 1

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass