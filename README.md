# Computer Vision for In-Store Customer Analysis for Kaikaku

## Overview
The tasks of this project were as follows: to become familiar with the Yolov8 computer vision model and to utilize its built-in capabilities to create a dataset that can distinguish between customers and employees. Customers are relatively simple for a computer program to distinguish from workers, as all workers wear a green hat. After attempting automatic labeling of ‘person’ objects wearing green hats, I achieved limited results. Consequently, I manually labeled around 441 image frames from CCTV footage and mirrored them to obtain a training set of 674 images. 

The online try-out features of Roboflow allowed me to gauge the accuracy of the model, and I found that the dataset size delivered sufficient results to continue. Subsequently, I wrote a simple program to track the customer’s time in the store, using a certain margin of error to determine the time tracked for each customer. As a result, I was able to calculate the average time a customer spent in the store. Furthermore, I modified the algorithm to utilize the polygon feature in Yolov8 development, which is useful for understanding how many potential customers may only linger by the entrance but decide not to enter. 

## 1. The Dataset

### 1.1 Training and Results
The dataset was created using Roboflow and Yolov8. The dataset proved to be more efficient than originally presumed, even given the limited amount of training data. 

It achieved the following accuracies in training:
- Customer Detection with 88% accuracy.
- Worker Detection with 97% accuracy.
- Overall Accuracy: 92%.

These accuracies are visualized in the following training graphs.

<img width="720" alt="Screenshot 2024-06-06 at 14 51 18" src="https://github.com/leopold16/computer_vision/assets/123328956/01d967d0-2496-4b50-8ff9-6e92f7d51423">

## 1.2 Testing the Dataset and Examples

The testing results provided highly accurate labeling of 'persons' as either 'workers' (in purple) or 'customers' (in yellow). In the provided image, we can observe the respective accuracies of the estimates per frame, which tend to be mostly above 75%.

<img width="418" alt="Screenshot 2024-06-06 at 14 51 42" src="https://github.com/leopold16/computer_vision/assets/123328956/9221174c-5aaa-4b79-ad64-5cca9ebff32e">

Similarly, displaying a video as a heatmap gave insights into the location within the store that customers frequent the most.

<img width="412" alt="Screenshot 2024-06-06 at 14 52 22" src="https://github.com/leopold16/computer_vision/assets/123328956/2e3bc5d3-166e-40a0-a5eb-1ee6e3b08b24">

## 1.3 Average Time Spent in Store 

The first algorithm developed attempts to make sense of the customer's time spent in the store. After conceptual analysis, the team decided that the algorithm was not beneficial for thorough analysis. Mainly, brief entries into the store (eg. delivery workers or potential customers peeking in) would alter the averages drastically. 

## 1.4 Conversion Rates 

As such, the idea of the conversion rate algorithm was born. I created two polygons that separated the viewable area into two different polygons. The 'entrance' polygon would keep track of the customers who entered the store. The 'store' polygon would keep track of the customers that entered the store from the 'entrance' area and thus successfully 'converted' into the store. 

Notice how the upper left corner shows the conversion rates. This is highly critical in analyzing the percentage of potential customers that enter the store.
Increasing the percentage of conversion would automatically result in increased sales and customer traffic. 

<img width="427" alt="Screenshot 2024-06-06 at 16 08 12" src="https://github.com/leopold16/computer_vision/assets/123328956/a809e201-5d26-4812-8e8e-fbbca6bb74d8">







