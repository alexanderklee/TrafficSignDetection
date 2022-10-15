import sys, signal, os
from time import sleep

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
    
    # Setup some directories (images, repsonses) in cwd
    imgDir, respDir = helper.setupDirs()
    
    # Need to define signal handler within the main func
    # to ensure variable state (or scope) is accessible
    # inside the signal function
    def handler(signum, frame):
        print()
        print("Ctrl-C Caught, exiting program now")
        
        ##############################################
        # Commenting file/directory removal for now  #
        # in the case the logs are needed even after #
        # a SIGINT is called                         #
        ############################################## 
        # print("Cleaning up images/ and repsonses/directories ..")      
        # # Remove all images in the images/ directory
        # files = os.listdir(imgDir)
        # for file in files: 
        #     pathfile = imgDir + file
        #     print("  Deleting images: ", pathfile)
        #     os.remove(pathfile)
        
        # # Delete images/ directory
        # print("  Deleting images/ directory: ", imgDir)
        # os.rmdir(imgDir)
        
        # # Remove all files in the responses/ directory
        # files = os.listdir(respDir)
        # for file in files: 
        #     pathfile = respDir + file
        #     print("  Deleting log file: ", pathfile)
        #     os.remove(pathfile)
        
        # # Delete responses/ directory
        # print("  Deleting responses/ directory: ", respDir)
        # os.rmdir(respDir)
        
        sys.exit(1)
        
    # Register signal event
    signal.signal(signal.SIGINT, handler)
    
    # Validate inputted key
    try:
        projList = client.projects()
        pName = menu.display(projList)
    except ScaleUnauthorized as err:
        print(err.message)
        sys.exit(1)
    
    # # Setup some directories (images, repsonses) in cwd
    # imgDir, respDir = helper.setupDirs()
     
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
            # Print current Task ID being worked on
            # (note: A newline is produced in tqdm because of the code below )
            # (      The progression of the bar looks sorta nice. You can    )
            # (      remove the two lines below if you want to see a single  )
            # (      self-updating progress bar                              )
            sys.stdout.write(" Current Task ID: " + task.task_id + '%\r')
            sys.stdout.flush()  
            # print(f" Current Task ID: {task.task_id:s}", end="\r")
            
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
                flagsDict["blurFlag"] = True
        
            # Iterate over all annotations
            for annotation in task.response['annotations']:
                # Flag missing Labels
                if not annotation['label']:
                    flagsDict["labelsFlag"] = True
                    
                # Flag 50% or Higher occlusions due to bad annotation or bad image
                of = helper.s2f(task.response['annotations'][0]['attributes']['occlusion'])
                if (of >= occ_thresh):
                    flagsDict["occFlag"] = True
                
                # Flag 50% or Higher truncation due to bad annotation or bad image
                tf = helper.s2f(task.response['annotations'][0]['attributes']['truncation'])
                if (tf >= trunc_thresh):
                    flagsDict["truncFlag"] = True
                
                # Big bounding box with respect to image area possible bad annotation or
                # bad sample image
                # Note: assumption the width and height data from Task are in pixels
                bbox_area = annotation['width'] * annotation['height']
                bbox_coverage = (bbox_area / imgArea) * 100
                if (bbox_coverage >= 0.2):
                    flagsDict["bigbboxFlag"] = True
                
                # High number of labels, more prone to annotator error
                if (len(task.response['annotations']) >= 10):
                    flagsDict["highlabelFlag"] = True
                
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
                        flagsDict["bboxoverlapFlag"] = True
                        # print(f"calc_iou(bbox[{box1idx}], bbox_dict[{box2idx}]) = {score}")
            
            # Process test flags and write out responses file
            flags.processFlags_dump_to_file(flagsDict, task.task_id, imgName, respFile)
            
            sleep(0.1)    
            pbar.update(1)
            
    helper.displayMsg()

if __name__ == "__main__":
    main()