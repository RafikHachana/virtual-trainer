import mediapipe as mp
import numpy as np
import cv2

from pose_estimation import base_estimator
from pose_estimation.human_pose import HumanPose


class MediaPipeEstimator(base_estimator.BaseEstimator):
    def __init__(self):
        # super().__init__()

        self.pose = mp.solutions.pose.Pose(
            model_complexity=2,
            min_tracking_confidence=0.7
        )

        self.draw = mp.solutions.drawing_utils


    def find_pose(self, image, draw=False) -> HumanPose:
        result = self.pose.process(image)


        if not result.pose_landmarks:
            return None
        if draw:
            self.draw.draw_landmarks(image, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            for ind, lm in enumerate(result.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy =  int(lm.x*w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)


        landmarks = list(result.pose_landmarks.landmark)
        return HumanPose(
            left_shoulder=self._landmark_to_coordinate(landmarks[11]),
            right_shoulder=self._landmark_to_coordinate(landmarks[12]),
            left_elbow=self._landmark_to_coordinate(landmarks[13]),
            right_elbow=self._landmark_to_coordinate(landmarks[14]),
            left_wrist=self._landmark_to_coordinate(landmarks[15]),
            right_wrist=self._landmark_to_coordinate(landmarks[16]),
            left_hip=self._landmark_to_coordinate(landmarks[23]),
            right_hip=self._landmark_to_coordinate(landmarks[24]),
            left_knee=self._landmark_to_coordinate(landmarks[25]),
            right_knee=self._landmark_to_coordinate(landmarks[26]),
            left_ankle=self._landmark_to_coordinate(landmarks[27]),
            right_ankle=self._landmark_to_coordinate(landmarks[28]),

            left_shoulder_visibility=self._landmark_visibility(landmarks[11]),
            right_shoulder_visibility=self._landmark_visibility(landmarks[12]),
            left_elbow_visibility=self._landmark_visibility(landmarks[13]),
            right_elbow_visibility=self._landmark_visibility(landmarks[14]),
            left_wrist_visibility=self._landmark_visibility(landmarks[15]),
            right_wrist_visibility=self._landmark_visibility(landmarks[16]),
            left_hip_visibility=self._landmark_visibility(landmarks[23]),
            right_hip_visibility=self._landmark_visibility(landmarks[24]),
            left_knee_visibility=self._landmark_visibility(landmarks[25]),
            right_knee_visibility=self._landmark_visibility(landmarks[26]),
            left_ankle_visibility=self._landmark_visibility(landmarks[27]),
            right_ankle_visibility=self._landmark_visibility(landmarks[28])
        )

    @staticmethod
    def _landmark_to_coordinate(landmark):
        return np.array([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    @staticmethod
    def _landmark_visibility(landmark):
        return landmark.visibility
