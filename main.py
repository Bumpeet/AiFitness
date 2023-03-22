import mediapipe as mp
import cv2
import numpy as np
from utils import *
from model import *

# 11, 23, 25, 27



def fitness_algo(img):
    global fitness

    res = fitness.process(img)

    if res.pose_landmarks:
        req_res = fitness.acqurie_marks(res)
        Lshoulder = req_res["Lshoulder"]
        Rshoulder = req_res["Rshoulder"]
        nose = req_res["nose"]

        offset_angle = angle((Lshoulder.x, Lshoulder.y), (nose.x, nose.y), (Rshoulder.x, Rshoulder.y))

        if offset_angle > fitness.offset_threshold:
            fitness.deal_inactive_offset()
            return fitness.image
  
        else:
            angles = fitness.calculate_angles(req_res)
            fitness.update_state()
            # fitness.deal_inactive_count()
            return fitness.image

    else:
        pass


def main():

    global fitness

    state_threshold =  {"s1":(0, 32),
                        "s2":(35, 65),
                        "s3":(75, 95)}
    feeback_threshold = {"f1": (0, 20),
                        "f2": (45, 180),
                        "f3": (50, 80),
                        "f4": (30, 180),
                        "f5": (95, 180)}
    offset_threshold = 10 # angle between nose, lShoulder and rShoulder
    inactive_threshold = 50
            
    fitness = AiFitness(state_threshold, feeback_threshold, offset_threshold, inactive_threshold)
    
    vid = r"squats_trim.mp4"
    cap = cv2.VideoCapture(vid)
    result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         15, (1280, 720))


    while(cap.isOpened()):
        ret, frame = cap.read()
        img = fitness_algo(frame)

        # cv2.imshow("window", img)
        result.write(img)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    
if __name__=="__main__":
    main()




