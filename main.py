import cv2
from pose_estimation.mediapipe_estimator import MediaPipeEstimator
from pose_estimation.human_pose_evolution import HumanPoseEvolution
from rep_counter.rep_counter import RepCounter

from datetime import datetime
import matplotlib.pyplot as plt
import time
from exercise_classifier import predict_exercise

from utilities import video_to_human_pose_evolution


video_path = "./sample_videos/video_3.mp4"
reference_video_path = "./sample_videos/video_3.mp4"


evolution = video_to_human_pose_evolution(video_path)

timeseries = evolution.right_elbow_angle_evolution()
all_timeseries = evolution.get_all_angles()
counter = RepCounter(evolution)

all_timeseries, all_extrema, reps, length, range_info = counter.all_extrema()



# print(range_info)
exercise_name = predict_exercise(all_timeseries)

print("Detected exercise: " + exercise_name)
print(f"You did {reps} reps with an average rep duration of {length:.2f} seconds")

reference_evolution = video_to_human_pose_evolution(reference_video_path)

counter = RepCounter(reference_evolution)

_, _, _, reference_length, reference_range_info = counter.all_extrema()

length_diff_percentage = 100*(abs(length - reference_length)/reference_length)

feedback = f"""
FEEDBACK:
- On average, your reps were {length_diff_percentage:.2f}% {'slower' if length > reference_length else 'faster'} then the reference exercise video.
"""
print(feedback)


for k, v in all_timeseries.items():
    plt.plot(*list(zip(*v)))
    if k in all_extrema:
        plt.scatter(*list(zip(*all_extrema[k])))

plt.show()

