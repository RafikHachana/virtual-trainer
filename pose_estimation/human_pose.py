"""
Dataclass that encompasses the keypoints of a human pose
"""
from dataclasses import dataclass
from typing import List
import numpy as np

# TODO: Update the fields of the class to whatever keypoints schema we are using

@dataclass
class HumanPose:
    # Generic version with just a list
    raw_points: List
    # The datapoints for each pose point
    # TODO: Make these private and use properties to read them
    left_shoulder: np.ndarray
    right_shoulder: np.ndarray

    left_elbow: np.ndarray
    right_elbow: np.ndarray

    left_wrist: np.ndarray
    right_wrist: np.ndarray

    left_hip: np.ndarray
    right_hip: np.ndarray

    left_knee: np.ndarray
    right_knee: np.ndarray

    left_ankle: np.ndarray
    right_ankle: np.ndarray

    # left_wrist: np.ndarray
    # right_wrist: np.ndarray

    def __init__(self, points: List[float] = None, **kwargs):
        """
        Converts the list of points to an object (depends on the pose estimation algorithm)
        """
        # The raw version (pretty useless)
        if points is not None:
            self.raw_points = points

        else:
            self.right_elbow = kwargs["right_elbow"]
            self.right_wrist = kwargs["right_wrist"]
            self.right_shoulder = kwargs["right_shoulder"]
            self.right_hip = kwargs["right_hip"]
            self.right_knee = kwargs["right_knee"]
            self.right_ankle = kwargs["right_ankle"]

            self.left_elbow = kwargs["left_elbow"]
            self.left_wrist = kwargs["left_wrist"]
            self.left_shoulder = kwargs["left_shoulder"]
            self.left_hip = kwargs["left_hip"]
            self.left_knee = kwargs["left_knee"]
            self.left_ankle = kwargs["left_ankle"]

            self.right_elbow_visibility = kwargs.get("right_elbow_visibility")
            self.right_wrist_visibility = kwargs.get("right_wrist_visibility")
            self.right_shoulder_visibility = kwargs.get("right_shoulder_visibility")
            self.right_hip_visibility = kwargs.get("right_hip_visibility")
            self.right_knee_visibility = kwargs.get("right_knee_visibility")
            self.right_ankle_visibility = kwargs.get("right_ankle_visibility")

            self.left_elbow_visibility = kwargs.get("left_elbow_visibility")
            self.left_wrist_visibility = kwargs.get("left_wrist_visibility")
            self.left_shoulder_visibility = kwargs.get("left_shoulder_visibility")
            self.left_hip_visibility = kwargs.get("left_hip_visibility")
            self.left_knee_visibility = kwargs.get("left_knee_visibility")
            self.left_ankle_visibility = kwargs.get("left_ankle_visibility")

    # Bunch of utility functions
    def _angle(self, first_point_name, middle_point_name, second_point_name) -> float:
        """
        Calculates the angles formed by the given 3 keypoints

        Returns:
            - angle (float): The angle in radians
        """
        # TODO: Solve the issue with the 3D angles
        # Get the vectors
        # x = self.__dict__[first_point_name][:2]- self.__dict__[middle_point_name][:2]
        # y = self.__dict__[second_point_name][:2] - self.__dict__[middle_point_name][:2]

        x = self.__dict__[first_point_name]- self.__dict__[middle_point_name]
        y = self.__dict__[second_point_name] - self.__dict__[middle_point_name]

        # print(x)
        # print(y)

        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)

        # angle = np.arctan2(np.linalg.det([x, y]), np.dot(x, y))
        angle = np.arccos(np.clip(np.dot(x, y), -1.0, 1.0))
        return angle

    # TODO: Add more functions for specific angles
    @property
    def right_shoulder_angle(self):
        return self._angle("right_elbow", "right_shoulder", "right_hip")

    @property
    def left_shoulder_angle(self):
        return self._angle("left_elbow", "left_shoulder", "left_hip")

    @property
    def right_elbow_angle(self):
        return self._angle("right_wrist", "right_elbow", "right_shoulder")

    @property
    def left_elbow_angle(self):
        return self._angle("left_wrist", "left_elbow", "left_shoulder")

    @property
    def right_hip_angle(self):
        return self._angle("right_shoulder", "right_hip", "right_knee")

    @property
    def left_hip_angle(self):
        return self._angle("left_shoulder", "left_hip", "left_knee")

    @property
    def right_knee_angle(self):
        return self._angle("right_hip", "right_knee", "right_ankle")

    @property
    def left_knee_angle(self):
        return self._angle("left_hip", "left_knee", "left_ankle")

    
    # TODO: We can use distance to check if the back is straight
    # This is less useful when using 3D points    
    def _distance(self, first_point_name, second_point_name):
        x = self.__dict__[first_point_name]
        y = self.__dict__[second_point_name]

        return np.linalg.norm(x-y)