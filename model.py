import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List
from time import perf_counter
import time

from utils import *



class AiFitness():
    def __init__(self, 
                 state_threshold, 
                 feedback_threshold, 
                 offset_threshold, 
                 inactive_threshold):
        
        self.state_threshold = state_threshold
        self.feedback_threshold = feedback_threshold
        self.offset_threshold = offset_threshold
        self.inactive_threshold = inactive_threshold

        self.offset_count = 0
        self.offset_count_threshold = 100
        self.state_count_threshold = 50
        self.state = 1
        self.state_sequence = [None, None, None]
        self.counter = counter()
        self.track_count = [0, 0, 0]
        # self.inactive_time = 0

        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
        self.image = None
        
    def process(self, image: np.ndarray) -> Dict[str, None]:
        self.image = image
        self.image.flags.writeable = False
        cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(self.image)
        self.image.flags.writeable = True
        cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        return results
    
    def acqurie_marks(self, results):
        req_results = {}
        req_results["nose"] = results.pose_landmarks.landmark[0]
        req_results["Lshoulder"] = results.pose_landmarks.landmark[11]
        req_results["Rshoulder"] = results.pose_landmarks.landmark[12]
        req_results["hip"] = results.pose_landmarks.landmark[23]
        req_results["knee"] = results.pose_landmarks.landmark[25]
        req_results["ankle"] = results.pose_landmarks.landmark[27]
        self.addCircle(req_results)
        self.addLines(req_results)
        return req_results
    
    def deal_inactive_offset(self):
        self.offset_count += 1
        self.counter.reset() if self.offset_count > self.offset_threshold else None
        self.addCounterToImg()
    
    def calculate_angles(self, req_results: tuple):
        self.offset_count = 0
        Lshoulder = req_results["Lshoulder"]
        hip = req_results["hip"]
        knee = req_results["knee"]
        ankle = req_results["ankle"]

        hip_shoulder = angle((hip.x, 0), (hip.x, hip.y), (Lshoulder.x, Lshoulder.y))
        knee_hip = angle((hip.x, hip.y), (knee.x, knee.y), (hip.x, 0))
        ankle_knee = angle((knee.x, knee.y), (ankle.x, ankle.y), (knee.x, 0))

        return (hip_shoulder, knee_hip, ankle_knee)

    def update_state(self, angles):
        (hip_shoulder, knee_hip, ankle_knee) = angles

        if self.state_threshold["s1"][0] <= knee_hip < self.state_threshold["s1"][1]:
            self.state = TrackState.s1
        elif self.state_threshold["s2"][0] <= knee_hip < self.state_threshold["s2"][1]:
            self.state = TrackState.s2
        elif self.state_threshold["s3"][0] <= knee_hip < self.state_threshold["s3"][1]:
            self.state = TrackState.s3

        self.update_counter()

    def update_counter(self):
        state_val = self.state-1
        self.update_other_count(state_val)

        if self.track_count[state_val] < self.state_count_threshold:
            if self.track_count[state_val] == 1:
                self.state_sequence.pop(0)
                self.state_sequence.append(self.state)
                cv2.putText(self.image, f'state: S{self.state}', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0,0))
                self.addCounterToImg()
        else:
            self.counter.reset()
            cv2.putText(self.image, f'state: S{self.state}', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0,0))
            self.addCounterToImg()

        
        if self.state_sequence[0] == 3 and self.state_sequence[1] == 2 and self.state_sequence[0] ==1:
            self.counter.correct()
            self.addCounterToImg()
        elif self.state_sequence[0] == 1 and self.state_sequence[1] == 2 and self.state_sequence[0] ==1:
            self.counter.incorrect()
            self.addCounterToImg()

    
    def update_other_count(self, val):
        if val == 0:
            self.track_count[0] += 1
            self.track_count[1] = 0
            self.track_count[2] = 0

        elif val == 1:
            self.track_count[0] == 0
            self.track_count[1] += 1
            self.track_count[2] == 0

        elif val == 2:
            self.track_count[0] == 0
            self.track_count[1] == 0
            self.track_count[2] += 1

    def deal_inactive_count(self):
        # track_count = np.array(self.track_count)
        # count = track_count[track_count!=0]
        self.counter.reset() if (self.track_count[self.state-1] > self.state_count_threshold) else None
        self.addCounterToImg()      

    def addCounterToImg(self) -> np.ndarray:

        cv2.putText(self.image, f"Correct: {self.counter.correct}", (1000, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
        cv2.putText(self.image, f"InCorrect: {self.counter.incorrect}", (1000, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    def addCircle(self, res: Dict[str, None]):
        h, w, _ = self.image.shape
         
        for key, val in res.items():
            x = int(val.x * w)
            y = int(val.y * h)
            cv2.circle(self.image, (x, y), 8, (255 , 255, 0), -1)

    def addLines(self, res):
        h, w, _ = self.image.shape
        angles = self.calculate_angles(res)

        x1, y1 = int(res["Lshoulder"].x * w),  int(res["Lshoulder"].y * h)
        x2, y2 =  int(res["hip"].x * w),  int(res["hip"].y * h)
        x3, y3 =  int(res["knee"].x * w),  int(res["knee"].y * h)
        x4, y4 = int(res["ankle"].x * w),  int(res["ankle"].y * h)

        lines = [(x1, y1),(x2, y2),(x3, y3),(x4, y4)]

        for i in range(len(lines)-1):
            cv2.line(self.image, lines[i], lines[i+1], (200, 200, 255), 6)
            cv2.line(self.image, lines[i+1], (lines[i+1][0], lines[i+1][1]-100), (100, 100, 255), 5)
            cv2.putText(self.image, f"{angles[0]}", (lines[i+1][0]+20, lines[i+1][1]+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

        cv2.ellipse(self.image, (x2, y2), (40, 40), 0, -90-angles[0], -90, (255, 255, 255), 3)
        cv2.ellipse(self.image, (x3, y3), (40, 40), 0, -90, -90+angles[1], (255, 255, 255), 3)
        cv2.ellipse(self.image, (x4, y4), (40, 40), 0, -90-angles[2], -90, (255, 255, 255), 3)



        
 
