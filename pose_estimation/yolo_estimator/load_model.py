# TODO: Add the YOLO repo as a submodule or some shit here


import torch
from torchvision import transforms
import sys
import os
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov7')))
# print(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov7')))

# print(sys.path)
from utils.datasets import letterbox
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np

model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov7/yolov7-w6-pose.pt'))

def load_model():
    device = torch.device("cpu")
    model = torch.load(model_path, map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    # if torch.cuda.is_available():
    #     # half() turns predictions into float16 tensors
    #     # which significantly lowers inference time
    #     model.half().to(device)
    return model

model = load_model()

def run_inference(image):
    # image = cv2.imread(url) # shape: (480, 640, 3)
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 768, 960])
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 768, 960])
    output, _ = model(image) # torch.Size([1, 45900, 57])
    return output, image


nc=model.yaml['nc'] # Number of Classes
nkpt=model.yaml['nkpt']


def visualize_output(output, image, draw=True):
    from utils.general import non_max_suppression_kpt
    print("YO")
    output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=nc, # Number of Classes
                                     nkpt=nkpt, # Number of Keypoints
                                     kpt_label=True)
    del non_max_suppression_kpt
    gc.collect()
    with torch.no_grad():
        output = output_to_keypoint(output)
    # nimg = image[0].permute(1, 2, 0) * 255
    # nimg = nimg.cpu().numpy().astype(np.uint8)
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    # print(output)
    # print(type(output))
    # print(output.shape)
    # # print(output[0, 7:].T)
    if draw:
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(image, output[idx, 7:].T, 3)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(nimg)
    # plt.show()

    # cv2.imshow("video", nimg)
    # cv2.waitKey(10)
    # We just take the first detected person here
    return output[0, 7:].T

# output, image = run_inference('index.jpeg') # Bryan Reyes on Unsplash
# visualize_output(output, image)
