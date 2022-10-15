import torch

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