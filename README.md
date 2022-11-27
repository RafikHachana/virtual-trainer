# Virtual Training Assistant using Pose Estimation

## Description

We present a Virtual Training Assistant using Computer Vision and Human Pose Estimation. Our final solution counts the number of exercise repetitions that were performed on an exercise, can generalize to any exercise or repetitive movement pattern, and also provides the user with feedback about the speed of their exercise execution by using a reference video that should also be supplied by the user. We have also implemented an exercise classifier that does exercise recognition using the keypoints of the human pose, however its performance was not satisfactory due to the lack of training data.



## How to run

You can just upload the Jupyter notebook `demo.ipynb` to Google Colab and run it there. All the needed demonstration videos will be downloaded. Keep in mind that the last part of the notebook (about YOLOv7) is a demo about the inefficiency of YOLO in this case, therefore it will crash your Colab runtime.
