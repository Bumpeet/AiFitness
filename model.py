import cv2
import mediapipe as mp
import numpy as np

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
        self.state_count = 0
        self.state_count_threshold = 20
        self.state = None
        self.state_list = []
        self.counter = counter

    def update():
        
        return None

        

