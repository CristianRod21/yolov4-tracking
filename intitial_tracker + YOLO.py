from __future__ import print_function
import sys
import cv2
from random import randint
import time
import tensorflow as tf
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
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        
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

        tmp_frame = cv2.resize(frame, (input_size, input_size))
        # Initialize Tracking
        tracker_type = "CSRT"

        create_time = time.time()
        multitracker = cv2.MultiTracker_create()
        info = "Tracker create time: %.2f ms" %(1000*(time.time()-create_time))
        print(info)
        image_h, image_w, _ = input_size,input_size,input_size
        print(image_h, image_w)
        #print(boxes.shape)
        #print(boxes[0])
        # Boxes of shape (1,50,4)
        add_tracker_time = time.time()


        for bbox in boxes[0]:
            bbox = tuple(bbox.numpy().tolist()) 
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

           
            #print('PRINTING BOUNDING BOXES USING YOLO AND OPENCV RECTANGLE ')
            #c1, c2 = (coords[1], coords[0]), (int(coords[1]+coords[3]), int(coords[0]+coords[2]))
            #color = (randint(0,255),randint(0,255),randint(0,255))
            #cv2.rectangle(frame, c1,c2,color,2,1)

            #test_bbox.append([coords[1], coords[1] + coords[3], coords[0], coords[0]+coords[2]] )
            if (coor[0]> 0 and coor[1]> 0 and coor[2]> 0 and coor[3] > 0 and coords[1] < 680):
                multitracker.add(createTrackerByName(tracker_type), tmp_frame, coords)
        info = "Tracker add time: %.2f ms" %(1000*(time.time()-add_tracker_time))
        print(info)
        #print(test_bbox)
        
        update_time = time.time()
        _, new_boxes = multitracker.update(tmp_frame)
        info = "Tracker update time: %.2f ms" %(1000*(time.time()-update_time))
        print(info)
        pred_bbox = [new_boxes, scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox_tracker(tmp_frame, pred_bbox)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        print(info)

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
