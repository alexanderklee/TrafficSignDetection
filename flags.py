import json

def processFlags_dump_to_file(flagsDict, task_id, imgName, respFile):
    quality_responses = {}
    quality_responses['TaskId'] = task_id
    quality_responses['ImageName'] = imgName
    
    if (flagsDict["blurFlag"]):
        quality_responses['Blur'] = 'WARN: Image could be blurry, please verify'
    else:
        quality_responses['Blur'] = 'NOTE: Images are not blurry'
        
    if (flagsDict["labelsFlag"]):
        quality_responses['MissingLabels'] = 'WARN: Missing labels present, please review'
    else:
        quality_responses['MissingLabels'] = 'NOTE: All labels present'
        
    if (flagsDict["occFlag"]):
        quality_responses['Occlusions'] = 'WARN: Images with >=50% occlusions present, please review'
    else:
        quality_responses['Occlusions'] = 'NOTE: Images are are within occlusion threshold'
    
    if (flagsDict["truncFlag"]):
        quality_responses['Truncation'] = 'WARN: Images with 50%> truncation present, please review'
    else:
        quality_responses['Truncation'] = 'NOTE: Images are within truncation threshold'
    
    if (flagsDict["bigbboxFlag"]):
        quality_responses['BigBbox'] = 'WARN: Bounding boxes >=20% of image area present, please review'
    else:
        quality_responses['BigBbox'] = 'NOTE: No large bounding boxes found'
        
    if (flagsDict["highlabelFlag"]):
        quality_responses['HighLabel'] = 'WARN: A high number of labels are present, verify quality'
    else:
        quality_responses['HighLabel'] = 'NOTE: Number of labels are within specification'
        
    if (flagsDict["bboxoverlapFlag"]):
        quality_responses['BboxOverlap'] = 'WARN: Bounding box overlap detected, verify quality'
    else:
        quality_responses['BboxOverlap'] = 'NOTE: No bounding box overlap detected'
    
    print("     Writing test responses to ", respFile)
    with open(respFile, 'w') as f:
        f.write(json.dumps(quality_responses, indent=4))