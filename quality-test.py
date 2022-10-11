import os, sys, argparse, pprint, json
from pickle import TRUE
import urllib.request
from shapely.geometry import box, Polygon
import torch
import cv2 

import nucleus
import scaleapi
from scaleapi.tasks import TaskReviewStatus, TaskStatus
from scaleapi.exceptions import ScaleException, ScaleUnauthorized, ScaleInvalidRequest

def calc_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    # iou = abs((poly1.intersection(poly2)).area / (poly1.union(poly2)).area - (poly1.intersection(poly2).area))
    iou = abs((poly1.intersection(poly2)).area / (poly1.union(poly2)).area)
    
    return(iou)

# Function to calculate IOU given a set of corner points
def calculate_iou(bpreds, bplabels):
    # bpreds shape is (N, 4) where N is the number of bboxes
    # blabels shape is (N, 4) where N is the number of bboxes
    box1_x1 = bpreds[..., 0:1]
    box1_y1 = bpreds[..., 1:2]
    box1_x2 = bpreds[..., 2:3] # Need tensor to be (N,1)
    box1_y2 = bpreds[..., 3:4] # If we don't then we'll end up with (N)
    
    box2_x1 = bplabels[..., 0:1]
    box2_y1 = bplabels[..., 1:2]
    box2_x2 = bplabels[..., 2:3]
    box2_y2 = bplabels[..., 3:4]
    
    # Get corner points of intersection
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # Calculate area of intersection by calculating
    # width and height of overlapping region. 
    # .clamp(0) is to cover non-overlapping edge cases
    # to avoid division with 0's. 
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)  
    
    # Calculate union of both boxes
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    
    # Calculate IOU and return
    # Note: need to substract intersection that area will be 
    #       counted twice as part of the union calculation. 
    #       1e-6 is used for numerical stability.
    return intersection / (box1_area + box2_area - intersection + 1e-6)

# A function to downloads image and returns image dimensions
def download_images_get_dims(imgUrl, imgDir):
    strSplit = imgUrl.split("/")
    imgName = strSplit[-1]
    saveDir = imgDir + imgName
    
    # Get image metadata and calculate blurriness
    urllib.request.urlretrieve(imgUrl, saveDir)
    print("     Saving task image to ", saveDir)
    imgData = cv2.imread(saveDir)
    height = imgData.shape[0]
    width = imgData.shape[1]
    gray = cv2.cvtColor(imgData, cv2.COLOR_BGR2GRAY)
    blur_level = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate area of the image
    area = height * width
    
    return(height, width, blur_level, area, saveDir, imgName)

# Helper function to convert a string to a float
def s2f(x):
    return float(x.strip('%'))

def setupDirs():
    imgDir = os.getcwd() + "/images/"
    respDir = os.getcwd() + "/responses/"
    os.makedirs(imgDir, exist_ok=True)
    os.makedirs(respDir, exist_ok=True)
    
    return(imgDir, respDir)

def processFlags_dump_to_file(blurFlag,labelsFlag, occFlag, truncFlag, bigbboxFlag, highlabelFlag, bboxoverlapFlag, task_id, imgName, respFile):
    quality_responses = {}
    quality_responses['TaskId'] = task_id
    quality_responses['ImageName'] = imgName
    
    if (blurFlag):
        quality_responses['Blur'] = 'WARN: Image could be blurry, please verify'
    else:
        quality_responses['Blur'] = 'NOTE: Images are not blurry'
        
    if (labelsFlag):
        quality_responses['MissingLabels'] = 'WARN: Missing labels present, please review'
    else:
        quality_responses['MissingLabels'] = 'NOTE: All labels present'
        
    if (occFlag):
        quality_responses['Occlusions'] = 'WARN: Images with >=50% occlusions present, please review'
    else:
        quality_responses['Occlusions'] = 'NOTE: Images are are within occlusion threshold'
    
    if (truncFlag):
        quality_responses['Truncation'] = 'WARN: Images with 50%> truncation present, please review'
    else:
        quality_responses['Truncation'] = 'NOTE: Images are within truncation threshold'
    
    if (bigbboxFlag):
        quality_responses['BigBbox'] = 'WARN: Bounding boxes >=20% of image area present, please review'
    else:
        quality_responses['BigBbox'] = 'NOTE: No large bounding boxes found'
        
    if (highlabelFlag):
        quality_responses['HighLabel'] = 'WARN: A high number of labels are present, verify quality'
    else:
        quality_responses['HighLabel'] = 'NOTE: Number of labels are within specification'
        
    if (bboxoverlapFlag):
        quality_responses['BboxOverlap'] = 'WARN: Bounding box overlap detected, verify quality'
    else:
        quality_responses['BboxOverlap'] = 'NOTE: No bounding box overlap detected'
    
    print("     Writing test responses to ", respFile)
    with open(respFile, 'w') as f:
        f.write(json.dumps(quality_responses, indent=4))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Please provide your API Key", type=str)
    parser.add_argument("--proj", required=True, help="Please provide your project name", type=str)
    return parser.parse_args()

def main():
    # Parse some CLI aruments
    args = get_args()
    client = scaleapi.ScaleClient(args.key)
    projname = args.proj
    
    # QA test threshold values
    iou_thresh = 0
    occ_thresh = 50
    trunc_thresh = 50
    
    # Test "LIVE" Key
    try:
        client.projects()
    except ScaleUnauthorized as err:
        print(err.message)
        sys.exit(1)
    
    # Setup some directories (images, repsonses) in cwd
    imgDir, respDir = setupDirs()
     
    # Retrieve the COMPLETED tasks from project
    try: 
        tasks = client.get_tasks(project_name = projname, status = TaskStatus.Completed)
    except ScaleInvalidRequest as err:
        print(err.message)
        sys.exit(1)
        
    
    # Iterate through project tasks and do some quality checks along the way
    for task in tasks:
        print("Working on task: ", task.task_id)
        
        # Initialize bbox dict to empty for each new task
        bbox_dict = {}
        bbox_idx = 0
        
        # Initialize flags to FALSE (assume the world is perfect)
        blurFlag=False
        labelsFlag=False
        occFlag=False
        truncFlag=False
        bigbboxFlag=False
        highlabelFlag=False
        bboxoverlapFlag=False
        
        # Create responses file per task_id
        respFile = respDir + task.task_id + ".json"
            
        # Flag possible blurry images and log image location on disk
        imgUrl = task.params['attachment']
        imgHeight, imgWidth, imgBlur, imgArea, saveDir, imgName = download_images_get_dims(imgUrl, imgDir)
        
        if imgBlur <= 400:
            blurFlag = TRUE
     
        # Iterate over all annotations
        for annotation in task.response['annotations']:
            # Flag missing Labels
            if not annotation['label']:
                labelsFlag = TRUE
                
            # Flag 50% or Higher occlusions due to bad annotation or bad image
            of = s2f(task.response['annotations'][0]['attributes']['occlusion'])
            if (of >= occ_thresh):
                occFlag = TRUE
            
            # Flag 50% or Higher truncation due to bad annotation or bad image
            tf = s2f(task.response['annotations'][0]['attributes']['truncation'])
            if (tf >= trunc_thresh):
                truncFlag = TRUE
            
            # Big bounding box with respect to image area possible bad annotation or
            # bad sample image
            # Note: assumption the width and height data from Task are in pixels
            bbox_area = annotation['width'] * annotation['height']
            bbox_coverage = (bbox_area / imgArea) * 100
            if (bbox_coverage >= 0.2):
                bigbboxFlag = TRUE
            
            # High number of labels, more prone to annotator error
            if (len(task.response['annotations']) >= 10):
                highlabelFlag=TRUE
            
            # Storing bounding box coords to be used for overlapping bbox tests
            # Assuming "left" and "top" is the top-left bbox corner (0,0)
            # Top-left
            x1 = annotation['left']
            y1 = annotation['top']
            # Top-right
            x2 = annotation['left'] + annotation['width']
            y2 = annotation['top']
            # Bottom-right
            x3 = annotation['left'] + annotation['width']
            y3 = annotation['top'] - annotation['height']
            # Bottom-left
            x4 = annotation['left']
            y4 = annotation['top'] - annotation['height']
            
            # Store instance of bbox into a dictionary of lists
            bbox_dict[bbox_idx] = [[x1,y1], [x2,y2],[x3,y3],[x4,y4]]
            bbox_idx += 1
        
            # Flag images with bright spots
            # (Could not get to this)
            
            # Flag images that are too dark or too bright
            # (Could not get to this)
            
            # Flag images where the color space are too similar (eg., one or two colors)
            # (Could not get to this)
        
        # Check for overlapping bounding boxes
        # Need two loops in order to provide iou function with two bbox coords
        for box1idx, (key, value) in enumerate(bbox_dict.items()):
            # print("box1idx:" ,box1idx)
            for box2idx in range(box1idx+1,len(bbox_dict.keys())):
                score = calc_iou(bbox_dict[box1idx], bbox_dict[box2idx])
                if (score > iou_thresh):
                    bboxoverlapFlag=TRUE
                    # print(f"calc_iou(bbox[{box1idx}], bbox_dict[{box2idx}]) = {score}")
        
        # Process test flags and write out responses file
        processFlags_dump_to_file(blurFlag,
                                  labelsFlag,
                                  occFlag,
                                  truncFlag,
                                  bigbboxFlag,
                                  highlabelFlag,
                                  bboxoverlapFlag,
                                  task.task_id,
                                  imgName,
                                  respFile)

if __name__ == "__main__":
    main()