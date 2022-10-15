import sys, time
from pickle import TRUE

# My modules
import iou, img, helper, flags, menu

# Progress bar
from tqdm import tqdm

# Scale.ai modules
import scaleapi
from scaleapi.tasks import TaskStatus
from scaleapi.exceptions import ScaleUnauthorized, ScaleInvalidRequest
    
def main():
    # Parse some CLI aruments
    args = helper.get_args()
    client = scaleapi.ScaleClient(args.key)
    
    # QA test threshold values
    iou_thresh = 0
    occ_thresh = 50
    trunc_thresh = 50
    
    # Validate inputted key
    try:
        projList = client.projects()
        pName = menu.display(projList)
    except ScaleUnauthorized as err:
        print(err.message)
        sys.exit(1)
    
    # Setup some directories (images, repsonses) in cwd
    imgDir, respDir = helper.setupDirs()
     
    # Retrieve the COMPLETED tasks from the user-specified project
    try: 
        tasks = client.get_tasks(project_name = pName, status = TaskStatus.Completed)
    except ScaleInvalidRequest as err:
        print(err.message)
        sys.exit(1)
        
    try: 
        task_count = client.get_tasks_count(project_name = pName, status = TaskStatus.Completed)
    except ScaleInvalidRequest as err:
        print(err.message)
        sys.exit(1)
    
    # Iterate through project tasks and do some quality checks along the way
    with tqdm(total=task_count, desc='Processing Tasks') as pbar:
        for task in tasks:
            # print("Working on task: ", task.task_id)
            
            # Initialize bbox dict to empty for each new task
            bbox_dict = {}
            bbox_idx = 0
            
            # Initialize flags to FALSE (assume the world is perfect)
            flagsDict = helper.initFlagsToFalse()
            
            # Create responses file per task_id
            respFile = respDir + task.task_id + ".json"
                
            # Flag possible blurry images and log image location on disk
            imgUrl = task.params['attachment']
            imgHeight, imgWidth, imgBlur, imgArea, saveDir, imgName = img.download_images_get_dims(imgUrl, imgDir)
            
            if imgBlur <= 400:
                flagsDict["blurFlag"] = TRUE
        
            # Iterate over all annotations
            for annotation in task.response['annotations']:
                # Flag missing Labels
                if not annotation['label']:
                    flagsDict["labelsFlag"] = TRUE
                    
                # Flag 50% or Higher occlusions due to bad annotation or bad image
                of = helper.s2f(task.response['annotations'][0]['attributes']['occlusion'])
                if (of >= occ_thresh):
                    flagsDict["occFlag"] = TRUE
                
                # Flag 50% or Higher truncation due to bad annotation or bad image
                tf = helper.s2f(task.response['annotations'][0]['attributes']['truncation'])
                if (tf >= trunc_thresh):
                    flagsDict["truncFlag"] = TRUE
                
                # Big bounding box with respect to image area possible bad annotation or
                # bad sample image
                # Note: assumption the width and height data from Task are in pixels
                bbox_area = annotation['width'] * annotation['height']
                bbox_coverage = (bbox_area / imgArea) * 100
                if (bbox_coverage >= 0.2):
                    flagsDict["bigbboxFlag"] = TRUE
                
                # High number of labels, more prone to annotator error
                if (len(task.response['annotations']) >= 10):
                    flagsDict["highlabelFlag"] = TRUE
                
                # Storing bounding box coords to be used for overlapping bbox tests
                # Assuming "left" and "top" is the top-left bbox corner (0,0)
                # (Top-left)
                x1 = annotation['left']
                y1 = annotation['top']
                # (Top-right)
                x2 = annotation['left'] + annotation['width']
                y2 = annotation['top']
                # (Bottom-right)
                x3 = annotation['left'] + annotation['width']
                y3 = annotation['top'] - annotation['height']
                # (Bottom-left)
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
                    score = iou.calc(bbox_dict[box1idx], bbox_dict[box2idx])
                    if (score > iou_thresh):
                        flagsDict["bboxoverlapFlag"] = TRUE
                        # print(f"calc_iou(bbox[{box1idx}], bbox_dict[{box2idx}]) = {score}")
            
            # Process test flags and write out responses file
            flags.processFlags_dump_to_file(flagsDict, task.task_id, imgName, respFile)
            
            time.sleep(0.1)    
            pbar.update(1)
            
    helper.displayMsg()

if __name__ == "__main__":
    main()