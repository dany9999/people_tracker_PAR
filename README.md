
#  Artificial Vision – People Tracking and Pedestrian Attribute Recognition

## Introduction

This project, developed for the **Artificial Vision** course at the **University of Salerno**, aims to design and implement a complete system for **people detection, tracking, and attribute recognition** within video streams.

The system processes a video input to:
- Detect all the people present in the scene;  
- Track each individual’s movement over time;  
- Identify specific attributes such as:
  - Gender (male/female);  
  - Presence of accessories (bag, hat);  
  - Upper and lower clothing colors (11 color classes);  
- Analyze people’s movement within defined regions of interest (ROIs) and measure their dwell time.

This integrated pipeline enables robust analysis of human activity and appearance — a task relevant for **smart surveillance, social robotics, business intelligence**, and **multi-camera re-identification**.

The project combines **YOLO-based object detection**, **DeepSORT tracking**, and a **custom ResNet50 model** fine-tuned on the **PETA dataset** to recognize pedestrian attributes. The architecture achieves real-time performance while maintaining high accuracy through optimized parameter settings and frame sampling strategies.

---

### Technologies and Tools
- **Python**
- **YOLOv8 / YOLOv5** for people detection  
- **DeepSORT** for multi-object tracking  
- **ResNet50** (fine-tuned on **PETA dataset**) for pedestrian attribute recognition  
- **OpenCV**, **PyTorch**, **NumPy**, **Matplotlib**
- **Visual Studio Code** for development and testing  
- **NVIDIA GeForce RTX 3070** GPU  

---

### System Pipeline Overview
1. **Frame Sampling** – Extracts frames at half the video’s framerate for faster processing.  
2. **Detection** – YOLOv8 detects all people in the current frame.  
3. **Tracking** – DeepSORT assigns consistent IDs across frames using motion and appearance features.  
4. **Attribute Recognition** – The ResNet50 model predicts gender, clothing color, and accessory presence.  
5. **Aggregation** – Each person’s most frequent attributes are stored as the final output.





## Introduction

This repository contains code for Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT) and Pedestrian Attribute Recognition.

NOTE: This project was based on the following repositories:

- https://github.com/Ayushman-Choudhuri/yolov5-deepsort
- https://github.com/hiennguyen9874/person-attribute-recognition
- https://github.com/nwojke/deep_sort



## Installation

First, clone the repository:

```
git clone https://github.com/dany9999/people_tracker_PAR
```

### Install requirements
```
pip3 install requirements.txt

```


### PAR model 
Then, download pre-trained PAR detector file from [here](https://drive.google.com/drive/folders/1Ya4gTu5hHhgN2-PptSpWzFleJB8oAJpt?usp=share_link) and insert it in **people_tracker_PAR/src/Pedestrian_attribute_Rec** 


## choice model detector

Insert in **config.yml** file the desired model:
```
 main:
  model_name: 'yolov8x'  <----
```


## Running the system

this example runs the system on a video and generates an output video and a file **results.txt** with the pedestrian attributes and people's interaction with the regions of interest:

```
python3 main.py --video video.mp4 --configuration config.txt --results results.txt
```

NOTE: In the **config.txt** file there are the coordinates of the ROIs


