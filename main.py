import mediapipe as mp
import cv2
import numpy as np
from utils import *

# 11, 23, 25, 27
offset_count = 0
state_count_threshold = 5
s1_count = 0
s2_count = 0 
s3_count = 0
state_list = []
correct = 0
incorrect = 0

state_threshold =  {"s1":(0, 32),
                    "s2":(35, 65),
                    "s3":(75, 95)}
feeback_threshold = {"f1": (0, 20),
                     "f2": (45, 180),
                     "f3": (50, 80),
                     "f4": (30, 180),
                     "f5": (95, 180)}
offset_threshold = 10
inactive_threshold = 5
counters = {"A":0, "I":0}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def process(image: np.ndarray):


    try:

        # image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        nose = results.pose_landmarks.landmark[0]
        Lshoulder = results.pose_landmarks.landmark[11]
        Rshoulder = results.pose_landmarks.landmark[12]
        hip = results.pose_landmarks.landmark[23]
        knee = results.pose_landmarks.landmark[25]
        ankle = results.pose_landmarks.landmark[27]


        if (Lshoulder.visibility >= 0.7 and hip.visibility >= 0.7 and knee.visibility >= 0.7 and ankle.visibility >= 0.7):
            if (angle((Lshoulder.x, Lshoulder.y),(nose.x, nose.y),(Rshoulder.x, Rshoulder.y)) > offset_threshold):
                offset_count = offset_count + 1
                if offset_count > 20:
                    correct = 0
                    incorrect = 0
                    return cv2.addText(image.copy(), f"correct:{correct}, incorrect:{incorrect}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))

                else:
                    return cv2.addText(image.copy(), f"correct:{correct}, incorrect:{incorrect}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))
                
            else:
                offset_count = 0

                hip_shoulder = angle((Lshoulder.x, Lshoulder.y), (hip.x, hip.y), (hip.x, 0))
                knee_hip = angle((knee.x, knee.y), (hip.x, hip.y), (hip.x, 0))
                ankle_knee = angle((ankle.x, ankle.y), (knee.x, knee.y), (knee.x, 0))

                if state_threshold["s1"][0] < knee_hip and knee_hip < state_threshold["s1"][1]:
                    s2_count = 0
                    s3_count = 0

                    frame_state = "s1"
                    s1_count += 1

                    if s1_count < state_count_threshold:

                        if state_list[-1]=="s1" and state_list[-2] == "s2" and state_list[-3] == "s3":
                            correct += 1
                        else:
                            incorrect += 1
                        
                        if s1_count == 1:
                            state_list.add(frame_state)

                        return cv2.addText(frame.copy(), f'state:{frame_state}', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))
                    
                    else:
                        correct = 0
                        incorrect = 0
                        return cv2.addText(image.copy(), f"correct:{correct}, incorrect:{incorrect}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))
                    
                elif state_threshold["s2"][0] < knee_hip and knee_hip < state_threshold["s2"][1]:
                    s1_count = 0
                    s3_count = 0

                    frame_state = "s2"
                    s2_count += 1

                    if s2_count < state_count_threshold:
                        if s2_count == 1:
                            state_list.add(frame_state)
                        return cv2.addText(frame.copy(), f'state:{frame_state}', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))
                    
                    else:
                        correct = 0
                        incorrect = 0
                        return cv2.addText(image.copy(), f"correct:{correct}, incorrect:{incorrect}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))
                    
                elif state_threshold["s3"][0] < knee_hip and knee_hip < state_threshold["s3"][1]:
                    s1_count = 0
                    s2_count = 0

                    frame_state = "s3"
                    s3_count += 1

                    if s3_count < state_count_threshold:
                        if s3_count == 1:
                            state_list.add(frame_state)
                        return cv2.addText(frame.copy(), f'state:{frame_state}', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))
                    
                    else:
                        correct = 0
                        incorrect = 0
                        return cv2.addText(image.copy(), f"correct:{correct}, incorrect:{incorrect}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))           

        else:
            state_list.clear()
            det_count += 1
            if det_count < state_count_threshold:
                return cv2.addText(image.copy(), f"correct:{correct}, incorrect:{incorrect}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))
            
            else:
                correct = 0
                incorrect = 0
                return cv2.addText(image.copy(), f"correct:{correct}, incorrect:{incorrect}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255))
        
        
    except Exception as e:
        pass
        


vid = r"squats_trim.mp4"
cap = cv2.VideoCapture(vid)

while(cap.isOpened()):
    ret, frame = cap.read()

    if frame is not None :

        frame = process(frame)
        # frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("after", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


