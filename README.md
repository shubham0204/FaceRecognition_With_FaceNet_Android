
![banner](https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/assets/41076823/5922415d-ff61-4276-9817-7e472ebac7c4)

# Face Recognition with FaceNet In Android

> A simple, efficient face detection app that allows us to add images of multiple persons in the device's internal storage, and recognize them in real-time through the camera-feed
 
### Features

* Reusable `FaceNet.kt` module that can easily used in other projects
* The app's design is simple and efficient, with focus of code readability and beginner-friendliness
* Face detection pipeline built in C++ for faster performance 
* Use of CameraX for configuring preview
* Using Jetpack Compose for UI 


![Working of the app](images/app_1.gif)



Given a cropped image of a human face, the FaceNet model produces a vector or a list of 128 elements termed as an *embedding*. The *embedding* is a compact representation of the human face given to it, and this vector when compared with the another vector, generated from some other face, can be used to determine if the two faces are of the same person. The FaceNet has been trained in manner which enables it to produce *similar* vectors for faces which belong to the same person. 

The task of identifying a person is reduced to comparing *face vectors* i.e. embeddings generated from the FaceNet model, using a suitable metric. In this project, we use the L2-norm and the cosine similarity. 



  
## Usage

![Intended File Structure](images/fig_2.png)
  
So, an user can store images in his/her device in a specific folder. If, for instance, the user wants the app to recognize  
two people namely "Rahul" and "Neeta". So the user needs to store the images by creating two directories namely "Rahul" and "Neeta"   
and store their images inside of these directories. For instance, the file structure for the working example ( as shown above in the GIF ),

![Intended File Structure](images/fig_4.png)

The app will then process these images and classify these people thereafter. For face recognition, Firebase MLKit is used which   
fetches bounding boxes for all the faces present in the camera frame.  
  
> For better performance, we recommend developers to use more images of the subjects, they need to recognize.

## Working  

![Sample Prediction](images/fig_3.png)
  
The app's working is described in the steps below:
  
1. Scan the `images` folder present in the internal storage. Next, parse all the images present within `images` folder and store   
the names of sub directories within `images`. For every image, collect bounding box coordinates ( as a `Rect` ) using MLKit.
   Crop the face from the image ( the one which was collected from user's storage ) using the bounding box coordinates.   
  
2. Finally, we have a list of cropped `Bitmap` of the faces present in the images. Next, feed the cropped `Bitmap` to the FaceNet   
model and get the embeddings ( as `FloatArray` ). Now, we create a `HashMap<String,FloatArray>` object where we store the names of   
the sub directories as keys and the embeddings as their corresponding values. 

The above procedure is carried out only on the app's startup. The steps below will execute on each camera frame.  
  
1. Using `androidx.camera.core.ImageAnalysis`, we construct a `FrameAnalyser` class which processes the camera frames. Now, for a   
given frame, we first get the bounding box coordinates ( as a `Rect` ) of all the faces present in the frame. Crop the face from   
the frame using these boxes.  
2. Feed the cropped faces to the FaceNet model to generate embeddings for them. We compare the embedding with a suitable metric and
form clusters for each user. We compute the average score for each cluster. The cluster with the best score is our output.
The final output is then stored as a `Prediction` and passed to the `BoundingBoxOverlay` which draws boxes and   
text.  
3. For multiple images for a single user, we compute the score for each image. An average score is computed for each group.
  The group with the best score is chosen as the output. See `FrameAnalyser.kt`.

```  
images ->  
    Rahul -> 
         image_rahul_1.png -> score=0.6 --- | average = 0.65 --- |
         image_rahul_2.png -> score=0.5 ----|                    | --- output -> "Rahul"
    Neeta ->                                                     |
         image_neeta_1.png -> score=0.4 --- | average = 0.35 --- |
         image_neeta_2.png -> score=0.3 ----|             
 ```

See [`FaceNetModel.kt`](https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/blob/master/app/src/main/java/com/ml/quaterion/facenetdetection/FaceNetModel.kt) and [`FrameAnalyser.kt`](https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/blob/master/app/src/main/java/com/ml/quaterion/facenetdetection/FrameAnalyser.kt) for the code.
  
## Limitations  
  
Predictions may go wrong as FaceNet does not always produce similar embeddings for the same person. 
Consider the accuracy of the FaceNet model while using it in your apps. In that case, you may learn to use the `FaceNetModel` class separating for using FaceNet in some other tasks.  
