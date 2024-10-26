# Assessing Car Damage Using Deep Learning Techniques
Car damage detection using Mask Regional Convolution Neural Network

# Project Introduction
- In this project I have developed a software that was able to detect four types of vehicle damages namely, Scratch, Dent, Dislocation and Shatter. Also I have provided a tentative repair cost for the type of damage detected. I have developed a website using Flask framework. Front end of the website is built using Bootstrap software.
- Automation is the next step in the automobile industry, and a software which is able to detect and localize damage in the car has various real world applications. In the software I have used Mask Regional Convolutional Network which is at the pinnacle of image detection techniques. Due to the complexity of the task I have used transfer learning to develop the software in three stages. Each stage added a new layer of complexity to the task, this helped the software prioritize learning the current task. I had also used Image Augmentation, which helped the model to not over fit the training data.

# Image Annotation

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/122675966-8302e900-d1f9-11eb-8623-3a94ac231d7a.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/122676009-9dd55d80-d1f9-11eb-99aa-7525630aa98b.png" />
</p>


# Stages

1) Stage 1- In stage 1, I trained the model on 'Damage' class and, used images annotated with bounding boxes.
2) Stage 2- In this stage training was done using four classes: damage-1 (scratch), damage-2 (dent), damage-3 (shatter) and damage-4(dislocation), in this stage I used images annotated with bounding boxes as well.
3) Stage 3- Stage 3 used images annotated with polygons, and four classes namely Scratch, Dent, Shatter and Dislocation. 

# Hyper Parameter Values
- You can use the following hyper parameter values to train your model. I found these values to give the best results.

- **LEARNING_RATE**: 0.001
- **LEARNING_MOMENTUM**: 0.9
- **WEIGHT_DECAY**: 0.0001
- **DETECTION_MIN_CONFIDENCE**: 0.8
- **STEPS_PER_EPOCH**: 100
- **NUM_CLASSES**: 5
- **MASK_POOL_SIZE**: 14
- **POOL_SIZE**: 7
- **VALIDATION_STEPS**: 50
- **RPN_ANCHOR_SCALES**: (16, 32, 64, 128, 256)
- **RPN_ANCHOR_RATIOS**: [0.5, 1, 1.5]
- **IMAGE_MIN_DIM**: 512
  
# Deployment

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/122676093-04f31200-d1fa-11eb-853e-202102a1b8e7.png" />
</p>


# Steps to Implement the Project
1) Put your_trained_weights.m5 file in the model directory.
2) Change line #22 in app/utils.py to the name of the weights of your model.
3) Run main.py file and the website will be hosted on http://127.0.0.1:5000/. Following are the URL rules I developed, you can add or delete these rules according to your preference.
 ```python
app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/damageapp','damageapp',views.damageapp)
app.add_url_rule('/damageapp/damage','damage',views.damage,methods=['GET','POST'])
```
If you don’t want to use cost assessment functionality, just change cost_for_damage variable on line #45 of app/views.py to False. Cost assessment functionality is just for visual purposes and computes cost based on size of mask to size of image ratio.

# Reference

[1] matterport/Mask_RCNN. (2021). Retrieved 26 October 2024, from https://github.com/matterport/Mask_RCNN

