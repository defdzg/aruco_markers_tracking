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
from scipy.spatial.transform import Rotation as R

DICT = cv.aruco.DICT_4X4_50
REF_MARKER = 0
markersPose =  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

def draw_frame_markers(corners, ids, rejected, frame, mtx, dist):
    
    if len(corners) > 0:
        ids = ids.flatten()
        if REF_MARKER in ids:
            for (markerCorner, markerID) in zip(corners, ids):  
                if markerID < len(markersPose):
                    # Get individual marker corners
                    rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(markerCorner, 0.023, mtx, dist)
                    cv.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.01)
                    
                    # RPY angles
                    rotM = np.zeros(shape=(3,3))
                    cv.Rodrigues(rvec[0], rotM, jacobian = 0)
                    r = R.from_matrix(rotM)
                    angles = r.as_euler('zxy', degrees=True)
                    yaw = round(angles[0], 1)
                    roll = round(angles[1], 1)
                    pitch = round(angles[2], 1)
                    corners = markerCorner.reshape((4, 2))
                    
                    # Corners coordinates
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    
                    # Draw each marker corners
                    cv.line(frame, topLeft, topRight, (0, 255, 0), 2)
                    cv.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                    cv.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
                    
                    # Compute marker center coordinates
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv.putText(frame, 
                    ('id: ' + str(markerID)), 
                    (topLeft[0], topLeft[1] - 10), 
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2)
                    
                    markersPose[markerID] = [cX, cY, yaw]

            if len(ids) == len(markersPose):
            
                markersPose[1][0] -= markersPose[0][0]
                markersPose[1][1] -= markersPose[0][1]
                markersPose[2][0] -= markersPose[0][0]
                markersPose[2][1] -= markersPose[0][1]
                
                markersPose[0][0] = 0
                markersPose[0][1] = 0
                
                print(markersPose)

    return frame

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
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (width,height), 1, (width,height))
    
    # Initialize video source capture
    vs = cv.VideoCapture(src, apiPreference=cv.CAP_ANY, 
                          params=[
                              cv.CAP_PROP_FRAME_WIDTH, width,
                              cv.CAP_PROP_FRAME_HEIGHT,height,
                              cv.CAP_PROP_FPS, fps,
                              cv.CAP_PROP_AUTOFOCUS, 0 ])
    time.sleep(2.0)
    
    # Set ArUco dictionary and initalize parameters
    arucoDict = cv.aruco.Dictionary_get(dict)
    arucoParams = cv.aruco.DetectorParameters_create()

    while(True):
        
        _ , frame = vs.read()
        frame = cv.undistort(frame, mtx, dist, None, newcameramtx)
        
        # Convert current frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = cv.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
        
        # Draw current coordinates and makers data
        frame = draw_frame_markers(corners, ids, rejected, frame, mtx, dist)
        
        # Display current frame with it's drawing
        cv.imshow('frame', frame)
        
        key = cv.waitKey(1) & 0xFF
        # Quit capture if 'q' key is pressed 
        if key == ord('q'):
            vs.release()
            cv.destroyAllWindows()
            break

# Main
def main():

    # Load .env file camera variables
    load_dotenv("./camera.env")
    SRC = int(os.getenv("SRC"))
    WIDTH = int(os.getenv("WIDTH"))
    HEIGHT = int(os.getenv("HEIGHT"))
    FPS = int(os.getenv("FPS"))
    MODEL = os.getenv("MODEL")
    
    capture(MODEL, SRC, WIDTH, HEIGHT, FPS, DICT)

# Entrypoint
if __name__ == "__main__":
    
    main()
