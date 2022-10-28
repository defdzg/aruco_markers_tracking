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

def inversePerspective(rvec, tvec):
    
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

def draw_frame_markers(corners, ids, rejected, frame, mtx, dist):
    
    flag = 0
    
    if len(corners) > 0:
        
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            
            # Get individual marker corners
            rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(markerCorner, 0.023, mtx, dist)
            cv.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.01)
            rotM = np.zeros(shape=(3,3))
            cv.Rodrigues(rvec[0], rotM, jacobian = 0)
            r = R.from_matrix(rotM)
            angles = r.as_euler('zxy', degrees=True)
            y = round(angles[0], 1)
            r = round(angles[1], 1)
            p = round(angles[2], 1)
            corners = markerCorner.reshape((4, 2))
            
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Get each corner coordinates
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
            #cv.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            # Display current marker center coordinates
            cv.putText(frame, 
            ('id: ' + str(markerID) + ', x:' + str(cX) + ', y:' + str(cY) + ', r:' + str(r) + ', p:' + str(p) + ', y:' + str(y)), 
            (topLeft[0], topLeft[1] - 10), 
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2)
            
            # Measuring distance in px and drawing lines from the reference marker to the moving marker
            if (markerID == 2):
                flag = 1
                refcX = cX
                refcY = cY
            else:
                if(flag == 1):
                    cv.line(frame, (refcX,refcY), (cX, cY), (0, 255, 0), 2)
                    if (markerID == 1):
                        distanceX =  cX - refcX
                        distanceY = refcY - cY
                        cv.putText(frame, ('Dist1-2: (' + str(distanceX) + ',' + str(distanceY) + ') px'),
                                   (1600, 70), 
                                    cv.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2)
                    if (markerID == 0):
                        distanceX =  cX - refcX
                        distanceY = refcY - cY
                        cv.putText(frame, ('Dist0-2: (' + str(distanceX) + ',' + str(distanceY) + ') px'),
                                   (1600, 100), 
                                    cv.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2)

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

    # Initialize video source capture
    vs = cv.VideoCapture(src, apiPreference=cv.CAP_ANY, 
                          params=[
                              cv.CAP_PROP_FRAME_WIDTH, width,
                              cv.CAP_PROP_FRAME_HEIGHT,height,
                              cv.CAP_PROP_FPS, fps,
                              cv.CAP_PROP_AUTOFOCUS, 0 ])
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
        # Estimate detected markers 
        
        rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners, 0.04, mtx, dist);
        cv.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.01)

        #frame = draw_frame_markers(corners, ids, rejected, frame, mtx, dist)
        
        newcameramtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (width,height), 1, (width,height))
        frame = cv.undistort(frame, mtx, dist, None, newcameramtx)

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
