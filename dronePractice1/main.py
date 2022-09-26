import cv2
from utils import *
import keyboard

w, h = 640, 480

if __name__ == "__main__":
    myDrone = initTello()
    myDrone.takeoff()
    time.sleep(1)
    myDrone.streamon()
    cv2.nameWindow("drone")
    frame_read=myDrone.get_frame_read()
    time.sleep(2)

    while True:
        img = frame_read.frame
        cv2.imshow("Image", img)
        # Press "q" to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            myDrone.land()
            frame_read.stop()
            myDrone.streamoff()
            exit(0)
            break

        if keyboard.is_pressed('w'):
            # time.sleep(1)
            myDrone.move_forward(10)
        if keyboard.is_pressed('s'):
            # time.sleep(1)
            myDrone.move_back(10)
        if keyboard.is_pressed('a'):
            # time.sleep(1)
            myDrone.move_left(10)
        if keyboard.is_pressed('d'):
            # time.sleep(1)
            myDrone.move_right(10)

