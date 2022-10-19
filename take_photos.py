# take_photos.py

# Dependencies
import cv2 as cv
from datetime import datetime
import os
import time
from dotenv import load_dotenv

folder = "calibration_photos"

# Capture video to save photos for camera calibration procedure
def capture(src, width, height, fps):
    
    # Create folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Initialize video source capture
    vs = cv.VideoCapture(src, apiPreference=cv.CAP_ANY, 
                          params=[
                              cv.CAP_PROP_FRAME_WIDTH, width,
                              cv.CAP_PROP_FRAME_HEIGHT,height,
                              cv.CAP_PROP_FPS, fps])
    time.sleep(2.0)
    
    while(True):
        
        _ , frame = vs.read()
        cv.imshow('frame', frame)
        
        key = cv.waitKey(1) & 0xFF
        
        # Quit capture if 'q' key is pressed 
        if key == ord('q'):
            vs.release()
            cv.destroyAllWindows()
            break
        # Capture photo if 'c' key is pressed
        if key == ord('c'):
            # Save current frame in the desired repository
            cv.imwrite(os.path.join(folder, f"raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), frame)
    
    # Return folder directory where calibration fotos where saved        
    return folder
            

def main():
    
    # Load .env file camera variables
    load_dotenv("./camera.env")
    SRC = int(os.getenv("SRC"))
    WIDTH = int(os.getenv("WIDTH"))
    HEIGHT = int(os.getenv("HEIGHT"))
    FPS = int(os.getenv("FPS"))
    capture(SRC, WIDTH, HEIGHT, FPS)

# entrypoint
if __name__ == "__main__":
    
    main()



