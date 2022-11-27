import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from pose_estimation.human_pose_evolution import HumanPoseEvolution

def running_mean(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

class RepCounter:
    def __init__(self, evolution: HumanPoseEvolution):
        self.evolution = evolution

    def _derivative(self):
        timeseries = self.evolution.right_elbow_angle_evolution()

        result = []
        timestamps = []
        last_timestamp, last_value = timeseries[0]
        for timestamp, value in timeseries[1:]:
            d = (value - last_value)/(timestamp - last_timestamp)
            result.append(d)
            timestamps.append(timestamp)
            last_timestamp, last_value = timestamp, value

        return np.array(timestamps), np.array(result)

    def _count_reps(self, sorted_extrema, min_range, max_range, min_rep_duration=1):

        valid_extrema = sorted_extrema[:1]

        minima = None
        for timestamp, value in sorted_extrema[1:-1]:
            # We didn't figure out yet
            if minima is None:
                if value <= min_range:
                    minima = True
                    valid_extrema.append((timestamp, value))
                elif value >= max_range:
                    minima = False
                    valid_extrema.append((timestamp, value))
                continue

            if minima is True and value >= max_range:
                valid_extrema.append((timestamp, value))
                minima = False

            if minima is False and value <= min_range:
                valid_extrema.append((timestamp, value))
                minima = True

        valid_extrema.append(sorted_extrema[-1])

        
        reps = []
        counted = False
        for ind, (timestamp, value) in enumerate(valid_extrema):
            if ind + 2 >= len(valid_extrema):
                break

            if counted:
                counted = False
                continue
            
            # if valid_extrema[ind+2][0] - timestamp > min_rep_duration:
            reps.append(valid_extrema[ind+2][0] - timestamp)
            counted = True

        if len(reps) == 0:
            return valid_extrema, 0

        # print("Reps before time filtering", len(reps))

        average_length = sum(reps)/len(reps)

        rep_count = len(list(filter(lambda x: x > 0.75*average_length, reps)))

        new_average_length = sum(reps)/len(reps)

        return valid_extrema, rep_count, new_average_length

    

    def _extrema(self, timeseries):
        # timeseries = self.evolution.right_elbow_angle_evolution()

        timestamps, values = zip(*timeseries)
        values = np.array(values)

        sampling_rate = 1/(timestamps[1] - timestamps[0])
        # print("Sampling rate", sampling_rate)

        ma_window = int(sampling_rate)

        # Apply filter
        values = running_mean(values, ma_window)

        maxima = argrelextrema(values, np.greater)[0]
        minima = argrelextrema(values, np.less)[0]

        absolute_max = np.max(values)
        absolute_min = np.min(values)

        angle_range = absolute_max - absolute_min

        if angle_range < 0.65:
            return [], angle_range, 0, list(zip(timestamps, values.tolist())), 0
        
        




        result = []
        for x in maxima:
            result.append([timestamps[x], values[x]])

        for x in minima:
            result.append([timestamps[x], values[x]])
        # print(result)

        result.append([timestamps[-1], values[-1]])
        result.append([timestamps[0], values[0]])


        sorted_extrema = sorted(result, key=lambda x: x[0])

        valid_extrema, reps, average_length = self._count_reps(sorted_extrema, absolute_min + 0.3*(angle_range), absolute_max - 0.3*(angle_range))

        # print(f"You did {min(maxima.shape[0], minima.shape[0])} reps!")
        return valid_extrema, angle_range, reps, list(zip(timestamps, values.tolist())), average_length
        # return result, angle_range, min(maxima.shape[0], minima.shape[0]), zip(timestamps, values.tolist())

    def all_extrema(self):
        angles = self.evolution.get_all_angles()
        smoothed_results = {}
        angle_extrema = {}

        potential_rep_counts = {}

        range_information = {}

        reps_values = []
        # extrema
        for name, timeseries in angles.items():
            # print(f"Using {name}")
            extrema, range, reps, smoothed, rep_length = self._extrema(timeseries)


            if reps:
                reps_values.append(reps)

            smoothed_results[name] = smoothed
            if len(extrema):
                angle_extrema[name] = extrema

            if reps not in potential_rep_counts:
              potential_rep_counts[reps] = {
                  "score": 0,
                  "count": 0,
                  "length": 0 
              }

            potential_rep_counts[reps]["score"] += range
            potential_rep_counts[reps]["count"] += 1
            potential_rep_counts[reps]["length"] += rep_length

            range_information[name] = range

        # print("Potential REPS", reps_values)

        rep_scores = {}

        for k, v in potential_rep_counts.items():
          rep_scores[k] = v["score"]/v["count"]

        result = list(rep_scores.keys())[0]

        for k, v in rep_scores.items():
          if v > rep_scores[result]:
            result = k
            
        average_rep_length = potential_rep_counts[result]["length"]/potential_rep_counts[result]["count"]

        return smoothed_results, angle_extrema, result, average_rep_length, range_information

        
