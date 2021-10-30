# Object Detection 

As typical for object detectors, the features learned by the convolutional layers are passed onto a classifier which makes the detection prediction. In YOLO, the prediction is based on a convolutional layer that uses 1×1 convolutions.

YOLO is named “you only look once” because its prediction uses 1×1 convolutions; the size of the prediction map is exactly the size of the feature map before it.



## YOLOv1 :

**YOLOv1** is a single-stage object detection model. Object detection is framed as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

The network uses features from the entire image to predict each bounding box. It also predicts all bounding boxes across all classes for an image simultaneously. This means the network reasons globally about the full image and all the objects in the image.

## YOLOv3 :

**YOLOv3** (You Only Look Once, Version 3) is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. YOLO uses features learned by a deep convolutional neural network to detect an object.

In YOLOv3 and its other versions, the way this prediction map is interpreted is that each cell predicts a fixed number of bounding boxes. Then, whichever cell contains the center of the ground truth box of an object of interest is designated as the cell that will be finally responsible for predicting the object. There is a ton of mathematics behind the inner workings of the prediction architecture.

Reference : [https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection)
