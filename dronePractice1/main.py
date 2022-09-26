import cv2
from utils import *
import keyboard

w, h = 640, 480

if __name__ == "__main__":
    myDrone = initTello()
    myDrone.takeoff()
    while True:
        img = telloGetFrame(myDrone, w, h)
        cv2.imshow("Image", img)
        # Press "q" to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            time.sleep(1)
            myDrone.land()
            break

        if keyboard.is_pressed('w'):
            time.sleep(1)
            myDrone.move_forward(70)
        if keyboard.is_pressed('s'):
            time.sleep(1)
            myDrone.move_back(70)
        if keyboard.is_pressed('a'):
            time.sleep(1)
            myDrone.move_left(70)
        if keyboard.is_pressed('d'):
            time.sleep(1)
            myDrone.move_right(70)

