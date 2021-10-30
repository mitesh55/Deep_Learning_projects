# Image Captioning

 ![Image Captioning-img_2](https://github.com/mitesh55/Deep_Learning_projects/blob/main/Image%20Captioning%20(%20TensorFlow%20)/demo.jpeg)

**Image Captioning** is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions.

The dataset will be in the form [image → captions]. The dataset consists of input images and their corresponding output captions.

## Network Tropology 

 ![Topology]()

**Encoder**

The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder.

**Decoder**

The Decoder is a Recurrent Neural Network(RNN) which does language modelling up to the word level. The first time step receives the encoded output from the encoder and also the <START> vector.

 **Training**
 
The output from the last hidden state of the CNN(Encoder) is given to the first time step of the decoder. We set x1 =<START> vector and the desired label y1 = first word in the sequence. Analogously, we set x2 =word vector of the first word and expect the network to predict the second word. Finally, on the last step, xT = last word, the target label yT =<END> token.
During training, the correct input is given to the decoder at every time-step, even if the decoder made a mistake before.

 **Testing**
The image representation is provided to the first time step of the decoder. Set x1 =<START> vector and compute the distribution over the first word y1. We sample a word from the distribution (or pick the argmax), set its embedding vector as x2, and repeat this process until the <END> token is generated.
During Testing, the output of the decoder at time t is fed back and becomes the input of the decoder at time t+1
 
 ## Network Architecture 
 
 ![Archi]()
