from .load_model import visualize_output, run_inference

from pose_estimation import base_estimator
from pose_estimation.human_pose import HumanPose

import numpy as np

class YOLOEstimator(base_estimator.BaseEstimator):
    def __init__(self):
        # super().__init__()

        pass


    def find_pose(self, image, draw=False) -> HumanPose:

        # print(image)
        output, image = run_inference(image) # Bryan Reyes on Unsplash
        result = visualize_output(output, image, draw=draw)


        landmarks = []

        for i in range(17):
            landmarks.append(result[i*3:(i+1)*3])


        # landmarks = list(result.pose_landmarks.landmark)
        return HumanPose(
            left_shoulder=self._landmark_to_coordinate(landmarks[5]),
            right_shoulder=self._landmark_to_coordinate(landmarks[2]),
            left_elbow=self._landmark_to_coordinate(landmarks[3]),
            right_elbow=self._landmark_to_coordinate(landmarks[6]),
            left_wrist=self._landmark_to_coordinate(landmarks[4]),
            right_wrist=self._landmark_to_coordinate(landmarks[7]),
            left_hip=self._landmark_to_coordinate(landmarks[11]),
            right_hip=self._landmark_to_coordinate(landmarks[8]),
            left_knee=self._landmark_to_coordinate(landmarks[12]),
            right_knee=self._landmark_to_coordinate(landmarks[9]),
            left_ankle=self._landmark_to_coordinate(landmarks[13]),
            right_ankle=self._landmark_to_coordinate(landmarks[10])

            # left_shoulder_visibility=self._landmark_visibility(landmarks[11]),
            # right_shoulder_visibility=self._landmark_visibility(landmarks[12]),
            # left_elbow_visibility=self._landmark_visibility(landmarks[13]),
            # right_elbow_visibility=self._landmark_visibility(landmarks[14]),
            # left_wrist_visibility=self._landmark_visibility(landmarks[15]),
            # right_wrist_visibility=self._landmark_visibility(landmarks[16]),
            # left_hip_visibility=self._landmark_visibility(landmarks[23]),
            # right_hip_visibility=self._landmark_visibility(landmarks[24]),
            # left_knee_visibility=self._landmark_visibility(landmarks[25]),
            # right_knee_visibility=self._landmark_visibility(landmarks[26]),
            # left_ankle_visibility=self._landmark_visibility(landmarks[27]),
            # right_ankle_visibility=self._landmark_visibility(landmarks[28])
        )

    @staticmethod
    def _landmark_to_coordinate(landmark):
        return np.array([
            landmark
        ])

    @staticmethod
    def _landmark_visibility(landmark):
        return landmark.visibility

 