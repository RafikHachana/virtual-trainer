from dataclasses import dataclass
from pose_estimation.human_pose import HumanPose
from typing import List

@dataclass
class HumanPoseEvolution:
    timeseries: List[HumanPose]

    def __init__(self):
        self.timeseries = []
        self.timestamps = []

        self.angles_cache = None

        self.skip_first = 0

    def add(self, timestamp: float, pose: HumanPose):
        self.timeseries.append(pose)
        self.timestamps.append(timestamp)

    def right_elbow_angle_evolution(self):
        result = []

        for pose in self.timeseries:
            result.append(pose.right_elbow_angle)

        return zip(self.timestamps, result)

    def get_all_angles(self):
        # Check the cache to avoid recalculations
        if self.angles_cache is not None:
            return self.angles_cache

        result = {
            "right_elbow_angle": [],
            "left_elbow_angle": [],
            "right_shoulder_angle": [],
            "left_shoulder_angle": [],
            "right_hip_angle": [],
            "left_hip_angle": [],
            "right_knee_angle": [],
            "left_knee_angle": [],
        }

        visibilities = {
            "right_elbow_angle": [],
            "left_elbow_angle": [],
            "right_shoulder_angle": [],
            "left_shoulder_angle": [],
            "right_hip_angle": [],
            "left_hip_angle": [],
            "right_knee_angle": [],
            "left_knee_angle": [],
        }

        for pose in self.timeseries[self.skip_first:]:
            result["right_elbow_angle"].append(pose.right_elbow_angle)
            result["left_elbow_angle"].append(pose.left_elbow_angle)
            result["right_shoulder_angle"].append(pose.right_shoulder_angle)
            result["left_shoulder_angle"].append(pose.left_shoulder_angle)
            result["right_hip_angle"].append(pose.right_hip_angle)
            result["left_hip_angle"].append(pose.left_hip_angle)
            result["right_knee_angle"].append(pose.right_knee_angle)
            result["left_knee_angle"].append(pose.left_knee_angle)

        # ZIP all of the angles with the timestamps
        for k, v in result.items():
            result[k] = list(zip(self.timestamps[self.skip_first:], v))

        # TODO: Filter the low visibility stuff
        # for k, v in result.items():
        #     result[k] = list(filter(lambda x))

        self.angles_cache = result

        return result


