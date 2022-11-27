# import exercise_classifier.training.data_preprocessing

import cv2
from pose_estimation.mediapipe_estimator import MediaPipeEstimator
from pose_estimation.human_pose_evolution import HumanPoseEvolution
from rep_counter.rep_counter import RepCounter

from datetime import datetime
import matplotlib.pyplot as plt
import time
import gc
import sys
# video_path = "~/Downloads/TelegramDesktop/IMG_1942.MOV"
dataset_path = "../workout"

import glob

files = glob.glob(dataset_path+"/*/**")

VIDEO_OFFSET = 9

VIDEOS_PER_BATCH = 30

FIRST_VIDEO = VIDEO_OFFSET*VIDEOS_PER_BATCH

LAST_VIDEO = (VIDEO_OFFSET+1)*VIDEOS_PER_BATCH

# for video_path in files:
#     print(video_path)


print("Number of videos", len(files))
# frames = []
if FIRST_VIDEO >= len(files):
    exit()


dataset_dict = {
    "left_shoulder": [],
    "right_shoulder": [],

    "left_elbow": [],
    "right_elbow": [],

    "left_wrist": [],
    "right_wrist": [],

    "left_hip": [],
    "right_hip": [],

    "left_knee": [],
    "right_knee": [],

    "left_ankle": [],
    "right_ankle": [],
    "frame_id": [],
    "video_id": [],
    "exercise_label": []
}

video_id = FIRST_VIDEO
for video_path in files[FIRST_VIDEO:min(LAST_VIDEO, len(files))]:
    exercise_label = video_path.split("/")[-2]
    print(exercise_label)

    # import cv2
    estimator = MediaPipeEstimator()

    # evolution = HumanPoseEvolution()
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video_path)

    start_time = datetime.utcnow()
    frame_id = 0
    while True:
        success, img = cap.read()
        if not success:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = estimator.find_pose(img_rgb, draw=True)

        #     print(results.right_elbow_angle)

        if results is not None:
            timestamp = time.time()
            # evolution.add(timestamp, results)

            frame = {
                "pose": results,
                "frame_id": frame_id,
                "video_id": video_id,
                "label": exercise_label
            }
            frame_id += 1

            # Add to the dataset dict
            dataset_dict["left_shoulder"].append(frame["pose"].left_shoulder)
            dataset_dict["right_shoulder"].append(frame["pose"].right_shoulder)

            dataset_dict["left_elbow"].append(frame["pose"].left_elbow)
            dataset_dict["right_elbow"].append(frame["pose"].right_elbow)

            dataset_dict["left_wrist"].append(frame["pose"].left_wrist)
            dataset_dict["right_wrist"].append(frame["pose"].right_wrist)

            dataset_dict["left_hip"].append(frame["pose"].left_hip)
            dataset_dict["right_hip"].append(frame["pose"].right_hip)

            dataset_dict["left_knee"].append(frame["pose"].left_knee)
            dataset_dict["right_knee"].append(frame["pose"].right_knee)

            dataset_dict["left_ankle"].append(frame["pose"].left_ankle)
            dataset_dict["right_ankle"].append(frame["pose"].right_ankle)
            dataset_dict["frame_id"].append(frame["frame_id"])
            dataset_dict["video_id"].append(frame["video_id"])
            dataset_dict["exercise_label"].append(frame["label"])


        if cv2.waitKey(1) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

    cap.release()
    cv2.destroyAllWindows() # destroy all opened windows

    video_id += 1
    # del cv2
    gc.collect()

    # Calculate memory consumption
    # local_vars = list(locals().items())
    # for var, obj in local_vars:
    #     print(var, sys.getsizeof(obj))
    # print(sum(map(sys.getsizeof, list(zip(*local_vars))[1])))

    # if video_id == 100:
    #     exit()



# for frame in frames:
#     dataset_dict["left_shoulder"].append(frame["pose"].left_shoulder)
#     dataset_dict["right_shoulder"].append(frame["pose"].right_shoulder)

#     dataset_dict["left_elbow"].append(frame["pose"].left_elbow)
#     dataset_dict["right_elbow"].append(frame["pose"].right_elbow)

#     dataset_dict["left_wrist"].append(frame["pose"].left_wrist)
#     dataset_dict["right_wrist"].append(frame["pose"].right_wrist)

#     dataset_dict["left_hip"].append(frame["pose"].left_hip)
#     dataset_dict["right_hip"].append(frame["pose"].right_hip)

#     dataset_dict["left_knee"].append(frame["pose"].left_knee)
#     dataset_dict["right_knee"].append(frame["pose"].right_knee)

#     dataset_dict["left_ankle"].append(frame["pose"].left_ankle)
#     dataset_dict["right_ankle"].append(frame["pose"].right_ankle)
#     dataset_dict["frame_id"].append(frame["frame_id"])
#     dataset_dict["video_id"].append(frame["video_id"])
#     dataset_dict["exercise_label"].append(frame["label"])

import pandas as pd
df = pd.DataFrame.from_dict(dataset_dict)

df.to_csv(F"frames_dataset_{VIDEO_OFFSET}.csv")

