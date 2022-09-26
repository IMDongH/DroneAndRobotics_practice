from djitellopy import Tello
import cv2
import time

def initTello():
    myDrone = Tello()
    # UDP 통신이어서 드론과 노트북이 1:1로 연결되어있어야한다
    myDrone.connect()

    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0

    print("\n * Drone battery percentage : " + str(myDrone.get_battery())+ "%")
    myDrone.streamoff()

    return myDrone

def moveTello(myDrone):
    myDrone.takeoff()
    time.sleep(5)

    myDrone.move_back(50)
    time.sleep(5)

    myDrone.rotate_clockwise(360)
    time.sleep(5)
    myDrone.move_forward(50)
    time.sleep(5)

    myDrone.flip_right()
    time.sleep(5)
    myDrone.flip_left()
    time.sleep(5)

    myDrone.land()
    time.sleep(5)

