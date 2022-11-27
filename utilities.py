import cv2
from pose_estimation.mediapipe_estimator import MediaPipeEstimator
from pose_estimation.yolo_estimator import YOLOEstimator
from pose_estimation.human_pose_evolution import HumanPoseEvolution
from rep_counter.rep_counter import RepCounter

from datetime import datetime
import matplotlib.pyplot as plt
import time


import skvideo.io
import numpy as np


def video_to_human_pose_evolution(video_path, model="blaze_pose"):
    estimator = None
    if model == "blaze_pose":
        estimator = MediaPipeEstimator()
    elif model == "yolo":
        estimator = YOLOEstimator()
    else:
        raise ValueError("The pose model should be either 'blaze_pose' or 'yolo'")

    evolution = HumanPoseEvolution()

    cap = cv2.VideoCapture(video_path)

    frames = []


    while True:
        success, img = cap.read()
        if not success:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = estimator.find_pose(img_rgb, draw=True)

        #     print(results.right_elbow_angle)

        # frames.append(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        if results is not None:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if timestamp != 0:
                evolution.add(timestamp, results)


        if cv2.waitKey(1) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

    cap.release()
    cv2.destroyAllWindows() # destroy all opened windows

    # Saving pose video

    # skvideo.io.vwrite("video.mp4", np.array(frames))
    return evolution

def webcam_to_human_pose_evolution(video_path):
    estimator = MediaPipeEstimator()

    evolution = HumanPoseEvolution()

    cap = cv2.VideoCapture(0)

    start_time = datetime.utcnow()

    frames = []

    while True:
        success, img = cap.read()
        if not success:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000

        frames.append((timestamp, img_rgb))

        # cv2.imshow("video", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        # exit()

        # if (datetime.utcnow() - start_time).total_seconds() > 15:
        #     break

        if cv2.waitKey(1) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

    cap.release()
    cv2.destroyAllWindows() # destroy all opened windows

    print("Finding 3D Pose ...")
    for timestamp, img_rgb in frames:
        results = estimator.find_pose(img_rgb, draw=True)

        #     print(results.right_elbow_angle)

        if results is not None:
            evolution.add(timestamp, results)

    return evolution