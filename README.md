# people_tracker_PAR


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

### PAR model 
Then, download pre-trained PAR detector file from here: 

--file google drive----

and insert it in people_tracker_PAR/src/Pedestrian_attribute_Rec 


## choice model detector

Insert in config.yml file the desired model:
```
 main:
  model_name: 'yolov8x'  <----
```


## Running the system

this example runs the system on a video and generates an output video and a file results.txt with the pedestrian attributes and people's interaction with the regions of interest:

```
python3 main.py --video video.mp4 --configuration config.txt --results results.txt
```

NOTE: In the config.txt file there are the coordinates of the ROIs


