import numpy as np
import math

def angle(left: tuple, middle: tuple, right: tuple) -> float:
    left = np.array(left)
    right = np.array(right)
    middle = np.array(middle)

    lVec = left - middle
    rVec = right - middle

    theta_rad = np.arccos(np.dot(lVec, rVec)/(np.linalg.norm(lVec, 2)*np.linalg.norm(rVec, 2)))
    theta_deg = np.rad2deg(theta_rad)

    return theta_deg



