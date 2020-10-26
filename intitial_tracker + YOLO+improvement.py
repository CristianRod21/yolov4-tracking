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


def update_tracker(trackers_dict,frame, newBboxes ):
    del_items = []
    for idx, items in enumerate(trackers_dict.items()):
        
        obj,tracker = items
        ok, bbox = tracker.update(frame)
        if (ok):
            # Updates bbox
            newBboxes[idx] = bbox
        else:
            print('Failed to track ', obj)
            del_items.append(obj)
    
    # Deletes if failed to track
    for idx, item in enumerate(del_items):            
        trackers_dict.pop(item)
        newBboxes.pop(idx)
    
    return newBboxes
        
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

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
    boxes, scores, classes, valid_detections  = get_detections(frame, input_size, infer)   
    newBboxes, trackers_dict, tmp_frame =update_tracked_objects(frame, input_size, boxes.numpy(),scores,classes,valid_detections)
        
    # Frames
    prev_frame_time = 0
    new_frame_time = 0
    
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
        boxes,scores,classes,valid_detections,newBboxes, trackers_dict
        
        allowed_classes = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'train']
        
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        
        # Resizes to yolo size
        new_frame = cv2.resize(frame, (input_size, input_size))
        # If its the first frame, or every 10th frames update the detector
        if (frame_id % 10 == 0 ):
            print(f"Frame {frame_id}, updating detections")
            # Updates bboxes from YOLO
            boxes, scores, classes, valid_detections  = get_detections(frame, input_size, infer)
            
            
            # delete detections that are not in allowed_classes
            deleted_indx = []
            
            for i in range(len(boxes[0])): 
                #print(f'Classes at {}')
                class_indx = int(classes[0][i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
            print(len(deleted_indx))
            print(boxes.shape, scores.shape)

            bboxes = np.expand_dims(np.delete(boxes[0], deleted_indx, axis=0), axis=0)
            scores = np.expand_dims(np.delete(scores[0], deleted_indx, axis=0), axis=0)
            classes = np.expand_dims(np.delete(classes[0], deleted_indx, axis=0), axis=0)
            print(bboxes.shape, scores.shape)
            
            # Updates the objects to be tracked
            newBboxes, trackers_dict, tmp_frame = update_tracked_objects(new_frame, input_size, bboxes,scores,classes,valid_detections)

            
        # Updates the tracked objects
        newBboxes = update_tracker(trackers_dict,new_frame, newBboxes )

        #for b in newBboxes:
            #if(b[0] > 0 and b[1]>0 and b[2]>0 and b[3]>0):
                #print(f'x1,x2 = ({b[0]},{b[0]+b[3]}) y y1,y2=({b[1]}, {b[3]+b[1]})')
                #crop=new_frame[b[0]:b[0]+b[2], b[1]:b[1]+b[3]]
                #cv2.imshow("Image", crop)
                #cv2.waitKey(0)

       
        # Pack and print the bbox
        pred_bbox = [newBboxes, scores, classes, valid_detections.numpy()]
        image = utils.draw_bbox_tracker(new_frame, pred_bbox)
        
        # TODO: Apply non-maxima supression?
        
        # Prints FPS
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time 
        fps = str(int(fps)) 
        cv2.putText(result, fps, (70, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA)
        
        #gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        #edged = cv2.Canny(gray, 50, 100)
        #edged = cv2.dilate(edged, None, iterations=1)
        #edged = cv2.erode(edged, None, iterations=1)
        #cv2.imshow("Image", edged)
        #cv2.waitKey(0)
        
                # find contours in the edge map
        #cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            #cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)
        #sort the contours from left-to-right and initialize the bounding box
        #point colors
        #(cnts, _) = contours.sort_contours(cnts)
        #colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

        #loop over the contours individually
        #for (i, c) in enumerate(cnts):
            #if the contour is not sufficiently large, ignore it
            #if cv2.contourArea(c) < 100:
                #continue
            #compute the rotated bounding box of the contour, then
            #draw the contours
            #box = cv2.minAreaRect(c)
            #box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            #box = np.array(box, dtype="int")
            #cv2.drawContours(result, [box], -1, (0, 255, 0), 2)
            #show the original coordinates
            #print("Object #{}:".format(i + 1))
            #print(box)


            #rect = perspective.order_points(box)
            #show the re-ordered coordinates
            #print(rect.astype("int"))
            #print("")
            
                #loop over the original points and draw them
            #for ((x, y), color) in zip(rect, colors):
                #cv2.circle(image, (int(x), int(y)), 5, color, -1)
            #draw the object num at the top-left corner
            #cv2.putText(image, "Object #{}".format(i + 1),
                #(int(rect[0][0] - 15), int(rect[0][1] - 15)),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            #show the image
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)

        
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
