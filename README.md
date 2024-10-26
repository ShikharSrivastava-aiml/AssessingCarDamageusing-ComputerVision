# Assessing Car Damage Using Deep Learning Techniques
Car damage detection using Mask Regional Convolution Neural Network

# Project Introduction
In this project, I developed a comprehensive software solution designed to detect four specific types of vehicle damages: scratch, dent, dislocation, and shatter. This software not only identifies the type of damage but also provides a tentative repair cost for each detected damage type, adding value by offering actionable insights for vehicle repairs. The entire platform is hosted on a user-friendly website built using the Flask framework, while the front-end interface employs the Bootstrap framework to ensure responsive design and a seamless user experience.

The automotive industry is rapidly moving towards automation, with applications for damage detection software extending across various real-world scenarios such as insurance assessments, rental car inspections, and personal vehicle maintenance. This software leverages advanced computer vision techniques to detect and localize damage on vehicles, thereby providing efficient and precise analysis that could benefit individuals, auto repair services, and insurance companies alike.

The core model used for damage detection is based on Mask Regional Convolutional Network (Mask R-CNN), a state-of-the-art approach renowned for its accuracy in image detection and segmentation. Mask R-CNN was chosen specifically for its ability to detect objects and distinguish between different areas of interest, which is essential when identifying varied types of vehicle damage. To handle the inherent complexity of this task, I implemented a transfer learning strategy and structured the training in three progressive stages. Each stage incrementally introduced additional layers of complexity, allowing the model to build a robust understanding of the specific task at hand. This staged approach helped the software efficiently prioritize the learning objectives, ensuring optimal results in damage detection.

Additionally, image augmentation techniques were employed to diversify the training data and mitigate overfitting, enabling the model to generalize well on unseen data. This augmentation step was critical, as it improved the model’s adaptability to various lighting conditions, angles, and backgrounds, which are common variables in real-world vehicle inspection scenarios.

# Stages and Optimal Hyperparameters

The model development followed a structured three-stage process, each stage progressively refining the model’s capability to detect and classify vehicle damages. In Stage 1, the model was trained on a general “Damage” class using images annotated with bounding boxes. This initial stage helped the model learn to recognize any visible damage on vehicles. Stage 2 introduced four specific damage types—damage-1 (scratch), damage-2 (dent), damage-3 (shatter), and damage-4 (dislocation). The training in this stage continued to utilize bounding box annotations, but the added classes allowed the model to begin distinguishing between different types of damage. In Stage 3, the model was trained with more refined polygonal annotations, while still using the four damage classes (scratch, dent, shatter, and dislocation). This last stage improved the precision of damage localization, making the model more adept at identifying the boundaries and areas of specific damage types.

To achieve optimal results, I used a set of fine-tuned hyperparameters. The learning rate was set to 0.001, with a learning momentum of 0.9 and weight decay at 0.0001, to ensure stable convergence without overfitting. The model’s confidence threshold for detection was set at detection_min_confidence = 0.8, and steps_per_epoch was set to 100, with validation_steps at 50 for efficient training and validation cycles. The number of classes was defined as 5, including the background class. Additionally, a mask pool size of 14 and pool size of 7 were chosen to manage the mask and bounding box dimensions. For the Region Proposal Network (RPN), anchor scales of (16, 32, 64, 128, 256) and anchor ratios of [0.5, 1, 1.5] were used to enhance multi-scale detection. Lastly, image_min_dim was set to 512, providing a balanced resolution for training without excessive memory usage. These hyperparameters collectively contributed to the model’s robustness and accuracy across various damage detection scenarios.

# Image Annotation

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/122675966-8302e900-d1f9-11eb-8623-3a94ac231d7a.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/122676009-9dd55d80-d1f9-11eb-99aa-7525630aa98b.png" />
</p>

# Deployment

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/122676093-04f31200-d1fa-11eb-853e-202102a1b8e7.png" />
</p>

# Steps to Implement the Project

To implement the vehicle damage detection software, follow these straightforward steps to ensure everything is set up correctly. First, place the file containing the trained model weights (your_trained_weights.m5) in the designated model directory. This file enables the software to load the pre-trained parameters necessary for accurate damage detection. Next, open the app/utils.py file, navigate to line 22, and modify it to reference the name of your specific model weights file, ensuring the program loads the correct model.

Once these preliminary adjustments are complete, run the main.py file to launch the application. This will host the website locally at http://127.0.0.1:5000/. The website’s structure includes several URL routes I developed to facilitate navigation and functionality; however, these can be customized according to preference. The following URL rules are preset within the application:

app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/damageapp','damageapp',views.damageapp)
app.add_url_rule('/damageapp/damage','damage',views.damage,methods=['GET','POST'])

The /base route serves as a template for the foundational elements of the website, while the / or index route directs users to the homepage. The /damageapp route opens the primary interface for damage assessment, and the final /damageapp/damage route activates the actual detection functionality, allowing users to submit images for damage analysis via GET or POST methods.

For those who prefer not to use the repair cost estimation feature, it is easy to disable this function. In app/views.py, on line 45, set the cost_for_damage variable to False. The cost estimation tool is mainly for visual purposes, calculating a tentative repair cost by assessing the ratio of the damage mask size to the overall image size.

# Reference

This project builds upon the Mask R-CNN architecture developed by Matterport, which is widely recognized for its efficacy in object detection and segmentation tasks. For more details on the Mask R-CNN framework, please refer to the official GitHub repository: matterport/Mask_RCNN (2021).

