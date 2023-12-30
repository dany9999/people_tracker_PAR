# people_tracker_PAR


# Bugs 

- write a control on img(succes) in while cap.isOpened(): 
'
results = object_detector.run_yolo(img)  # run the yolo v5 object detector 
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieleabbagnale/Desktop/people_tracker_PAR/src/YoloV5.py", line 37, in run_yolo
    frame_width = int(frame.shape[1]/self.downscale_factor)
'
