# aruco_detection.py

# python aruco_detection.py --dict DICT_6X6_50

# Dependencies
import cv2 as cv
import argparse
import json
import numpy as np
import os
import time
from dotenv import load_dotenv

# Existing ArUco dictionaries to be selected
ARUCO_DICT = {
    
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
 
}

# Read existing camera calibration parameters
def read_calibration(calibration_file):
    
    # Load .json file
    with open(calibration_file, 'r') as json_file:
        camera_data = json.load(json_file)
        dist = np.array(camera_data["dist"])
        mtx = np.array(camera_data["mtx"])
    return dist, mtx

def capture(model, src, width, height, fps, dict):
    
    # Read existing camera calibration parameters
    # mtx =  Camera matrix
    # dist  =  Distortion coefficients
    dist, mtx = read_calibration((model+".json"))

    # Initialize video source capture
    vs = cv.VideoCapture(src, apiPreference=cv.CAP_ANY, 
                          params=[
                              cv.CAP_PROP_FRAME_WIDTH, width,
                              cv.CAP_PROP_FRAME_HEIGHT,height,
                              cv.CAP_PROP_FPS, fps])
    time.sleep(2.0)
    
    # Set ArUco dictionary and initalize parameters
    arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[dict])
    arucoParams = cv.aruco.DetectorParameters_create()

    while(True):
        
        _ , frame = vs.read()
        # Convert current frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Detect ArUco markers
        corners, ids, rejected = cv.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
        # Estimate detected markers pose
        rvec, tvec, markerPointse = cv.aruco.estimatePoseSingleMarkers(corners, 0.023, mtx, dist);
        
        if len(corners) > 0:
            for i in range(0, len(ids)):
                
                # Draw IDs, frames and corners for each detected marker
                cv.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.01)
                cv.aruco.drawDetectedMarkers(frame, corners, ids) 
        
        # Display current fram with it's drawing
        cv.imshow('frame', frame)
        key = cv.waitKey(1) & 0xFF
        
        # Quit capture if 'q' key is pressed 
        if key == ord('q'):
            vs.release()
            cv.destroyAllWindows()
            break

# Set required script arguments
def arguments():
    
    parser = argparse.ArgumentParser(description='Generate .json camera correction settings after camera calibration.')
    parser.add_argument('--dict', type=str, required=True, help='Select desired ArUco dictionary')

    return parser.parse_args()

def main():
    
    # Set required arguments
    args = arguments()
    
    load_dotenv("./camera.env")
    
    SRC = int(os.getenv("SRC"))
    WIDTH = int(os.getenv("WIDTH"))
    HEIGHT = int(os.getenv("HEIGHT"))
    FPS = int(os.getenv("FPS"))
    MODEL = os.getenv("MODEL")
    
    capture(MODEL, SRC, WIDTH, HEIGHT, FPS, args.dict)

# entrypoint
if __name__ == "__main__":
    
    main()
