# Self-Driving Car Engineer Nanodegree
# Computer Vision
## Project: Vehicle Detection and Tracking

### Overview
The goals / steps of this project are the following:  

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.  
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images. 
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


### Final Result

The final pipleline has been successfully applied in [project video](https://youtu.be/5JIlVY1FgCk).  


### Histogram of Oriented Gradients (HOG)

1. HOG features.  
I started by reading in all the vehicle and non-vehicle images. Here is an example of some of the vehicle and non-vehicle classes:
![Training Samples](https://github.com/LevinJ/CarND-Advanced-Lane-Lines/blob/master/camera_calibration.png)

I then explored different color spaces and different skimage.hog() parameters (orientations, pixels_per_cell, and cells_per_block). I grabbed random images from each of the two classes and displayed them to get a feel for what the skimage.hog() output looks like.  

Here is an example using the LAB color space and HOG parameters of orientations=9, pixels_per_cell=(8, 8) and cells_per_block=(2, 2):  
![HOG Features](https://github.com/LevinJ/CarND-Advanced-Lane-Lines/blob/master/camera_calibration.png)



The code for part is contained in file preprocess/hogfeature.py and file preprocess/preprocess.preprocessdata.py. In partuclar, the LAB HOG feature extraction is done in PreprocessData::extract_features_labels
2. HOG parameters  

I tried differrent different orientations, pixels_per_cell,cells_per_block and settled on orientations=9, pixels_per_cell=(8, 8) and cells_per_block=(2, 2) as this combination has top performance. More importantly, this combinations allows me to conviently precompute hog features when slding windows over the image during detection process. Sliding windows stargegy is discussed below.


3. Classifier 

* After much exploration, I chose linear SVM for its good balance between speed and accuracy. This part is implemented in file implement/svmmodel.py  
* Features are scaled to zero mean and unit variance before training the classifier. sklearn StandardScaler is used to achieve this task.  
* Cross valdiation and grid search is used to find optimal hyper parameter for the models.  
* The validation accuracy of the final model is about 0.98 

### Sliding Window Search
1. Sliding window strategy
* We are using image pyramid and precomputed hog features to efficiently detect and locate cars
* Below is the break down
 1) scale the original 1280x720 image by different scales
 2) compute the hog feature for the scaled image
 3) sliding a 7x7 windows with stride 1 over the hog feature obrained above, which essentially the same as sliding a 64x64 over the scaled imge with a 8 pixles stride.
 4) predict on the 7x7 hog feature window (more precisely, it's 7x7x2x2x9)obtained
* The scale of image pyramid is determined by experimentations. the bottom line is to cover all the cars.
This part is mainly implmented in postprocess/pyramidhog.py, PyramidHog::get_window_feature, and postprocess/detectioninimage.py, DetectionInImage::process_image_RGB

2. Detection pipeline

The pipeline for vehicle detection in a single image is as below:
 1) with the sliding window search stragegy illustrated above, detect all the vehicles in the image
 2) use the confidence score of the predictions to filter some false positives
 3) use clustering (dbscan) to group detections
 4) if neccessary, do a second clustering to separate cars that are too close
 5) within the grouped detections, use the contour of the detections boxes as bouding box for the car

Here are some examples of test images to demonstrate the pipeline:
![Detection Pipleline](https://github.com/LevinJ/CarND-Advanced-Lane-Lines/blob/master/camera_calibration.png)
![Detection Pipleline](https://github.com/LevinJ/CarND-Advanced-Lane-Lines/blob/master/camera_calibration.png)

This part is implmented in postprocess/mergebox.py

### Video Implementation
1. Project video output
[project video](https://youtu.be/5JIlVY1FgCk)
2. False postive reduction
A heat map is generated to track the position of predicted vehicles in the past 10 frames. For the current frame, if the centroid of deteced vehicles is located in hot area of the heat map, we would consider that this detection as real, otherwise we would reject it. The use of heatmap reduce some false positives that occassionally pop up in the heat map.

This part is implemented in postprocess/frametracking.py, specifically FrameTracking::check_cars
3. Bouding box stabilization
The bounding box is very accurate in that it covers the majority part of the vechicles most of the time, but it jitters a lot from frame to frame. Moving average is used to stabilize the bounding box. the details are as below:
1. search its previous frame, and get the vehicle that is closet to it
2. check if its distance is less than a certain limit
3. if yes, apply moving average to current frame. new_box = 0.9 * last_bdbox + 0.1 * car_info['bdbox']

This part is implemented in postprocess/frametracking.py, specifically FrameTracking::__moving_average

### Discussion

This is a very interesting proejct in that it allows me to dive into the details of setting up a pipeline of object detection and tracking with a traditional computer vision approach. Current pipelines works well in the project video, but I can forsee it might not generalzie well in below scenarios:
* cars that are too small (less than 70x70)
* cars that are too close to each other
* some unseen negative samples in the scene

I think current pipeline can be further improved by introducing hard negative mining, and fine tuing of various parameters in the pipeline. Having said that, I think the deep learning end to end approach might have the potential to take this detection and trakcing performance to a whole new level. This is definitly in my to do list, with high priority. Can't wait to try it out:)

