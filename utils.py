import numpy as np
import math
import cv2

class counter:
    def __init__(self):
      self.correct = 0
      self.incorrect = 0
    
    def reset(self):
        self.correct = 0
        self.incorrect = 0

    def correct(self):
        self.correct += 1
    
    def inorrect(self):
        self.incorrect += 1


class TrackState:
    offset = 0
    s1 = 1
    s2 = 2
    s3 = 3

class TrackCount:
    def __init__(self) -> None:
        self.s1 = 0
        self.s2 = 0
        self.s3 = 0


def angle(left: tuple, middle: tuple, right: tuple) -> float:
    left = np.array(left)
    right = np.array(right)
    middle = np.array(middle)

    lVec = left - middle
    rVec = right - middle

    theta_rad = np.arccos(np.dot(lVec, rVec)/(np.linalg.norm(lVec, 2)*np.linalg.norm(rVec, 2)))
    theta_deg = np.rad2deg(theta_rad)

    return theta_deg


def addCounterToImg(image: np.ndarray, countr: counter) -> np.ndarray:

    cv2.putText(image, f"Correct: {countr.correct}", (1000, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    cv2.putText(image, f"InCorrect: {countr.incorrect}", (1000, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    return image
    



