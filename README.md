# YOLO V4 Tracking

YOLOv4, YOLOv4-tiny Implemented in Tensorflow 2.0 [1]. Combined with multi object tracking using OpenCV::Multitracker or DeepSort[3]. 

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

```
# Convert darknet weights to tensorflow
## yolov4
python save_model.py --weights /path/to/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 
```



## Usage:

You can use one of the trackers listed below (provided by opencv)

* BOOSTING
* MIL
* KCF
* TLD
* MEDIANFLOW
* GOTURN
* MOSSE
* CSRT

```
python "intitial_tracker + YOLO.py" --video path/to/video --model yolov4 --framework tensorflow --tracker_type [OPTION]  
```





For Deep Sort run:

```
python object_tracker.py --video /path/to/video --model yolov4   
```





## Acknowledgments

* [1] Hùng, V. (2020). tensorflow-yolov4-tflite. https://github.com/hunglc007/tensorflow-yolov4-tflite

* [2] Mallick, S. (2018). MultiTracker : Multiple Object Tracking using OpenCV (C++/Python). https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
* [3] Wotherspoon, J. (2020). yolov4-deepsort. https://github.com/theAIGuysCode/yolov4-deepsort