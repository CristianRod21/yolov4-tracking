import numpy as np
from time import time

# YOLO
newBboxes =  np.array([(167, 199, 26, 44), (147, 269, 41, 86), (209, 162, 18, 35), (90, 246, 51, 69), (204, 139, 12, 21), (131, 162, 21, 66), (206, 123, 11, 13), (171, 123, 10, 16), (310, 123, 13, 15), (157, 110, 8, 13), (181, 118, 8, 12), (285, 140, 27, 36), (295, 122, 12, 17), (89, 245, 53, 70), (130, 155, 23, 71), (287, 145, 23, 31), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)])

# Tracker
oldBoxes = np.array([(165.0, 199.0, 28.0, 44.0), (87.0, 247.0, 56.0, 74.0), (208.0, 167.0, 19.0, 30.0), (129.0, 161.0, 25.0, 69.0), (204.0, 138.0, 13.0, 22.0), (145.0, 264.0, 43.0, 95.0), (180.0, 117.0, 8.0, 13.0), (171.0, 124.0, 10.0, 16.0), (311.0, 122.0, 13.0, 17.0), (158.0, 111.0, 9.0, 12.0), (146.0, 265.0, 41.0, 92.0), (127.0, 158.0, 25.0, 70.0), (285.0, 143.0, 13.0, 38.0), (291.0, 145.0, 17.0, 32.0), (295.0, 123.0, 12.0, 18.0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)])

#height = y2-y1
#width = x2-x1
#x1,y1,height,width

#IOU format: x1,y1,x2,y2

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou

def get_max_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax
    
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

#For every new box calculates IOU to every other old box
for i, detectorBox in enumerate(newBboxes):
    max_iou_box_index = 0
    max_iou_box_score = 0
    for j,trackerBox in enumerate(oldBoxes):
        iou = get_iou_2(detectorBox, trackerBox ) 
        if (iou > 0.5):
            if (iou > max_iou_box_score ):
                max_iou_box_index = j
                max_iou_box_score = iou
    if (max_iou_box_score > 0.5):
        print(i, max_iou_box_index ,max_iou_box_score)

#print(f'New Bboxes {newBboxes}')
#print(f'Old Bboxes {oldBoxes}')

