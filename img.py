import urllib.request
import cv2

def download_images_get_dims(imgUrl, imgDir):
    strSplit = imgUrl.split("/")
    imgName = strSplit[-1]
    saveDir = imgDir + imgName
    
    # Get image metadata and calculate blurriness
    urllib.request.urlretrieve(imgUrl, saveDir)
    # print("     Saving task image to ", saveDir)
    imgData = cv2.imread(saveDir)
    height = imgData.shape[0]
    width = imgData.shape[1]
    gray = cv2.cvtColor(imgData, cv2.COLOR_BGR2GRAY)
    blur_level = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate area of the image
    area = height * width
    
    return(height, width, blur_level, area, saveDir, imgName)