humble-yolo is a minimal implementation of YOLO v1 I wrote to learn about the amazing YOLO algorithm.
Tutorial details:

https://medium.com/@ecaradec/humble-yolo-implementation-in-keras-64d1b63b4412
YOLO is a deep learning model that can predict object classes and location. It belongs to the group of classifications algorithm. 
# Data generation
Here we generate images with the texts "Varun" and "Kapoor" randomly placed inside an image. As we are generating images, we write bounding boxes of objects in a text file named the same way as their image.

# Installation Instructions
In order to use this program tensorflow and keras library are required. 
1. Download python3.8 version of [anaconda](https://www.anaconda.com/distribution/).
2. After downloading it, open the anaconda terminal and set the proxy settings to enable pip and conda installation via terminal.
3. Set up a virtual enviornment type this command at anaconda prompt: conda create -n tensorflowGPU pip python=3.8  
4. Activate the virtual enviornment: source activate tensorflowGPU
5. Install tensorflow and keras library by executing pip install keras tensorflow.

To test it run :

1. cd to the directory where this repo is clone by writing this in the terminal: cd humble-yolo
2. mkdir Labels, mkdir Images
1. python3.8 generate-dataset.py to generate data
2. python3.8 main.py --train --epoch 100 for training the network

You should see a list of images with bounding boxes. The first 10 images are test data not used for training. You can evaluate the performance of the network on those. The remaining images have been used for the training.

main.py saves weights when it complete training. If you want to run the network without training and just see the result, running main.py alone will load last weights and redisplay results.
