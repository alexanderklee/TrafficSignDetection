import os, argparse
from decimal import Decimal

# Helper function to convert a string to a float
def s2f(x):
    return float(x.strip('%'))

# Helper function to convert a string to a decimal
def s2d(y):
    return int(Decimal(y))

# Helper function to initialize a dict and values to FALSE
def initFlagsToFalse():
    flagsDict = {}
    flagsDict["blurFlag"]=False
    flagsDict["labelsFlag"]=False
    flagsDict["occFlag"]=False
    flagsDict["truncFlag"]=False
    flagsDict["bigbboxFlag"]=False
    flagsDict["highlabelFlag"]=False
    flagsDict["bboxoverlapFlag"]=False
    
    return(flagsDict)

def setupDirs():
    imgDir = os.getcwd() + "/images/"
    respDir = os.getcwd() + "/responses/"
    os.makedirs(imgDir, exist_ok=True)
    os.makedirs(respDir, exist_ok=True)
    
    return(imgDir, respDir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Please provide your API Key", type=str)
    
    # Using dynamic menu list instead of user-inputted project strings
    # parser.add_argument("--proj", required=True, help="Please provide your project name", type=str)
    
    return parser.parse_args()