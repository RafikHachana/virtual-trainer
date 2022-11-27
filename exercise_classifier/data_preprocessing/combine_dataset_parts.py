import pandas as pd

parts = 9

# for i in range(parts):

result = pd.concat([pd.read_csv(f"video_dataset_{i}.csv") for i in range(parts)])

result.to_csv("full_video_dataset.csv")
