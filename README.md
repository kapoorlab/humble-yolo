humble-yolo is a minimal implementation of YOLO v1 I wrote to learn about the amazing YOLO algorithm.
Tutorial details:

https://medium.com/@ecaradec/humble-yolo-implementation-in-keras-64d1b63b4412
To test it run :

1. cd to the directory where this repo is clone by writing this in the terminal: cd humble-yolo
2. mkdir Labels, mkdir Images
1. python3.8 generate-dataset.py to generate data
2. python3.8 main.py --train --epoch 100 for training the network

You should see a list of images with bounding boxes. The first 10 images are test data not used for training. You can evaluate the performance of the network on those. The remaining images have been used for the training.

main.py saves weights when it complete training. If you want to run the network without training and just see the result, running main.py alone will load last weights and redisplay results.
