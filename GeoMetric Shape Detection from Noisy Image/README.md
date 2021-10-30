# Geometric Shape Detection 

Here, We've made custom image dataset consists of noisy images with circle shape in it.

## Bounding Box Design

Here, model takes image (size=100,100) of noisy image consist circle at random place and of random size. Output of the model will be coordinates of polygon(rectangle).

**Approach**

In this problem, circle has been created inside the rectangular box of random size at random location in image.

**feature**

In this problem, image consist of circle inside rectangular box will be useful to build model

**Target**

Coordinates of rectangular box will be label [x,y,w,h]¶

**Loss Function**

Mean Squared Error is been used due to the nature of label (coordinates will be continous in nature)

**Implementation**

parameters = 19,09,396
Data Points = 18,000
