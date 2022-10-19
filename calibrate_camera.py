# calibrate_camera.py

# python calibrate_camera.py --new false

# Dependencies
import numpy as np
import cv2 as cv
from datetime import datetime
import argparse
import glob
import json
import os
from dotenv import load_dotenv
import take_photos

# Create .json file from the calibration camera results
def jsonify(model,ret, mtx, dist, rvecs, tvecs):
    
    # Create JSONEncoder object
    camera = {}
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    for variable in ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']:
        camera[variable] = eval(variable)

    # Create output .json file
    with open((model +".json"), 'w') as f:
        json.dump(camera, f, indent=4, cls=NumpyEncoder)

# Camera calibration procedure
def calibration(folder, model, rows, columns):
    
    # Termination criteria for the cornerSubPix() algorithm
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Ensure the number of rows and columns is one less than the actual rows and columns 
    # on your chessboard. The reason for doing this is that the algorithm will be looking 
    # for internal corners on the chessboard

    rows = rows - 1;
    columns = columns - 1;
    
    # Create some arrays to store the object points and image points from all the images of the chessboard
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    # Get the filenames of all the raw calibration images
    images = glob.glob(os.path.join(folder,'raw*.png'))
    print(len(images), "raw images found")

    for fname in images:
        
        # The image is converted to grayscale
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the cressboard corners
        chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv.findChessboardCorners(gray, (columns,rows), chessboard_flags)

        if ret == True:
            objpoints.append(objp)
            # Increase the accuracy using cornerSubPix()
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Uncomment if is desired to save Chessboard Corners images in the calibration photos directory
            #cv.drawChessboardCorners(img, (columns,rows), corners2, ret)
            # folder = "calibration_photos"
            #cv.imwrite(os.path.join(folder,f"corners_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), img)
            #cv.imshow('img', img)
            #cv.waitKey(1500)

    # Calibrate the camera using the corners that have been found
    # ret = RMS re-projection error
    # mtx =  Camera matrix
    # dist  =  Distortion coefficients
    # rvecs = Rotation vectors
    # tvecs = Translation vectors
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    cv.destroyAllWindows()
    
    #  Compute the arithmetical mean of the errors calculated for all the calibration images
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("Average re-projection: {}".format(mean_error/len(objpoints)) )
    
    # Create .json output file for the obtained calibration parameters
    jsonify(model, ret, mtx, dist, rvecs, tvecs)

# Set required script arguments
def arguments():
    
    parser = argparse.ArgumentParser(description='Generate .json camera correction settings after camera calibration.')
    parser.add_argument('--new', type=str, choices=['true', 'false'],required=True, help='Select if is required to take new photos for calibration')

    return parser.parse_args()

def main():
    
    # Set required arguments
    args = arguments()
    
    # Load .env file camera variables
    load_dotenv("./camera.env")
    MODEL = os.getenv("MODEL")
    BOARD_ROWS = int(os.getenv("BOARD_ROWS"))
    BOARD_COLUMNS = int(os.getenv("BOARD_COLUMNS"))
    
    # Take new photos
    if args.new == 'true':
        SRC = int(os.getenv("SRC"))
        WIDTH = int(os.getenv("WIDTH"))
        HEIGHT = int(os.getenv("HEIGHT"))
        FPS = int(os.getenv("FPS"))
        folder = take_photos.capture(SRC, WIDTH, HEIGHT, FPS)
        calibration(folder, MODEL, BOARD_ROWS, BOARD_COLUMNS)
    
    # Photos already exist
    else:
        folder = "calibration_photos"
        if os.path.exists(folder):
            calibration(folder, MODEL, BOARD_ROWS, BOARD_COLUMNS)
        else:
            print("There are no calibration photos to use. Run a new calibration procedure using the flag '--new true'.")    

# entrypoint
if __name__ == "__main__":
    
    main()