from shapely.geometry import box, Polygon

def calc(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    # iou = abs((poly1.intersection(poly2)).area / (poly1.union(poly2)).area - (poly1.intersection(poly2).area))
    iou = abs((poly1.intersection(poly2)).area / (poly1.union(poly2)).area)
    
    return(iou)