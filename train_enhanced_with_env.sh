#!/bin/bash
# Wrapper script to activate virtual environment and train

# Activate virtual environment
source /home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/YOLOv11-pt/YOLO/bin/activate

# Run training
./train_enhanced.sh
