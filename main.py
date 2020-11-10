from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D

from keras.layers.advanced_activations import LeakyReLU, PReLU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE-FILE_LOCKING"]="FALSE"
from keras import backend as K
from keras.models import load_model
import numpy as np
import sys
import cv2
import argparse
import csv
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt
import matplotlib.patches as patches


x_train = []
y_train = []

nb_boxes= 4
categories = 4
grid_w=1
grid_h=1
cell_w= 128
cell_h= 128
img_w=grid_w*cell_w
img_h=grid_h*cell_h

#
# Read input image and output prediction
#
def load_image(j):
    img = cv2.imread('Images/%d.PNG' % j)
#    img = cv2.resize(img,(64,64))
    
    x_t = img_to_array(img)

    y_t = []
    with open("Labels/%d.txt" % j, newline = '\n') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        for train_vec in reader:
               catarr = [float(s) for s in train_vec[0:categories]]
               xarr = [float(s) for s in train_vec[categories:]]
               newxarr = []
               for b in range(nb_boxes):
                   newxarr+= [xarr[s] for s in range(len(xarr))]
                   
               trainarr = catarr + newxarr    
               
               y_t.append(trainarr)
               
    
    return [x_t, y_t]

#
# Load all images and append to vector
# 
#for j in range(0, 10):
for j in range(10, 5000):
    [x,y] = load_image(j)
    x_train.append(x)
    y_train.append(y)

x_train = np.array(x_train)
y_train = np.array(y_train)

#
# Define the deep learning network
#

# model 2
i = Input(shape=(img_h,img_w,3))

x = Conv2D(16, (1, 1))(i)
x = Conv2D(32, (3, 3))(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(16, (3, 3))(x)
x = Conv2D(32, (3, 3))(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(256, activation='sigmoid')(x)
x = Dense(grid_w*grid_h*(categories+nb_boxes*5), activation='sigmoid')(x)
x = Reshape((grid_w*grid_h,(categories+nb_boxes*5)))(x)

model = Model(i, x)

#
# The loss function orient the backpropagation algorithm toward the best direction.
#It does so by outputting a number. The larger the number, the further we are from a correct solution.
#Keras also accept that we output a tensor. In that case it will just sum all the numbers to get a single number.
# 
# y_true is training data
# y_pred is value predicted by the network
def custom_loss(y_true, y_pred):
    # define a grid of offsets
    # [[[ 0.  0.]]
    # [[ 1.  0.]]
    # [[ 0.  1.]]
    # [[ 1.  1.]]]
    grid = np.array([ [[float(x),float(y)]]*nb_boxes   for y in range(grid_h) for x in range(grid_w)])

    # first three values are classes : cat, rat, and none.
    # However yolo doesn't predict none as a class, none is everything else and is just not predicted
    # so I don't use it in the loss
    y_true_class = y_true[...,0:categories]
    y_pred_class = y_pred[...,0:categories] 

    # reshape array as a list of grid / grid cells / boxes / of 5 elements
    pred_boxes = K.reshape(y_pred[...,categories:], (-1,grid_w*grid_h,nb_boxes,5))
    true_boxes = K.reshape(y_true[...,categories:], (-1,grid_w*grid_h,nb_boxes,5))
      
    # sum coordinates of center of boxes with cell offsets.
    # as pred boxes are limited to 0 to 1 range, pred x,y + offset is limited to predicting elements inside a cell
    y_pred_xy   = pred_boxes[...,0:2] +(grid)
    # w and h predicted are 0 to 1 with 1 being image size
    y_pred_wh   = pred_boxes[...,2:4]
    # probability that there is something to predict here
    y_pred_conf = pred_boxes[...,4]

    # same as predicate except that we don't need to add an offset, coordinate are already between 0 and cell count
    y_true_xy   = true_boxes[...,0:2]
    # with and height
    y_true_wh   = true_boxes[...,2:4]
    # probability that there is something in that cell. 0 or 1 here as it's a certitude.
    y_true_conf = true_boxes[...,4]

    clss_loss  = K.sum(K.square(y_true_class - y_pred_class), axis=-1)
    xy_loss    = K.sum(K.sum(K.square(y_true_xy - y_pred_xy),axis=-1)*y_true_conf, axis=-1)
    wh_loss    = K.sum(K.sum(K.square(K.sqrt(y_true_wh) - K.sqrt(y_pred_wh)), axis=-1)*y_true_conf, axis=-1)

    # when we add the confidence the box prediction lower in quality but we gain the estimation of the quality of the box
    # however the training is a bit unstable

    # compute the intersection of all boxes at once (the IOU)
    intersect_wh = K.maximum(K.zeros_like(y_pred_wh), (y_pred_wh + y_true_wh)/2 - K.square(y_pred_xy - y_true_xy) )
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
    true_area = y_true_wh[...,0] * y_true_wh[...,1]
    pred_area = y_pred_wh[...,0] * y_pred_wh[...,1]
    union_area = pred_area + true_area - intersect_area
    iou = intersect_area / union_area

    conf_loss = K.sum(K.square(y_true_conf*iou - y_pred_conf), axis=-1)

    # final loss function
    d = 2 * xy_loss + wh_loss + conf_loss + clss_loss
    
    if False:
        d = tf.Print(d, [d], "loss")
        d = tf.Print(d, [xy_loss], "xy_loss")
        d = tf.Print(d, [wh_loss], "wh_loss")
        d = tf.Print(d, [clss_loss], "clss_loss")
        d = tf.Print(d, [conf_loss], "conf_loss")
    
    return d

model = Model(i, x)



#
# Training the network
#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', help='train', action='store_true')
parser.add_argument('--epoch', help='epoch', const='int', nargs='?', default=1)
parser.add_argument('--learning_rate', help='learning_rate', const = 'double', nargs='?',default = 0.0001)
args = parser.parse_args()

if args.train:
    
    if os.path.exists('cordyolo.h5'):
    
        print('loading weights')
        model.load_weights('cordyolo.h5')
    

    adam = keras.optimizers.SGD(lr=float(args.learning_rate))
    model.compile(loss=custom_loss, optimizer=adam) # better
    print(model.summary())
    model.fit(x_train, y_train, batch_size=200, epochs=int(args.epoch))

    model.save_weights('cordyolo.h5')
else:
    model.load_weights('cordyolo.h5')

axes=[0 for _ in range(100)]
fig, axes = plt.subplots(5,5)

#
# Predict bounding box and classes for the first 25 images
#
for j in range(0,25):
    im = load_image(j)

    #
    # Predict bounding box and classes
    #
    img = cv2.imread('Images/%d.PNG' % j)
    #img = cv2.resize(img, (img_w,img_h))
    #data = img_to_array(img)
    P = model.predict(np.array([ img_to_array(img) ]))
 
    #
    # Draw each boxes and class score over each images using pyplot
    #
    col = 0
    for row in range(grid_w):
        for col in range(grid_h):
            p = P[0][col*grid_h+row]

            boxes = p[categories:].reshape(nb_boxes,5)
            clss = np.argmax(p[0:categories - 1])
            
            ax = plt.subplot(5,5,j+1)
            imgplot = plt.imshow(img)

            i = 0
            xmean = 0
            ymean = 0
            wmean = 0
            hmean = 0
            confmean = 0
            for b in boxes:
                x = b[0]+float(row)
                y = b[1]+float(col)
                w = b[2]
                h = b[3]
                conf = b[4]
                if conf < 0.5:
                   continue
                xmean = xmean + x
                ymean = ymean + y
                wmean = wmean + w
                hmean = hmean + h
                confmean = confmean + conf
            if confmean < 0.5:
               continue
            xmean = xmean/len(boxes)
            ymean = ymean/len(boxes)
            wmean = wmean/len(boxes)
            hmean = hmean/len(boxes)
            confmean = confmean/len(boxes)    
            color = ['r','g','b','0'][clss]
            rect = patches.Rectangle((xmean*cell_w-wmean/2*img_w, ymean*cell_h-hmean/2*img_h), wmean*img_h, hmean*img_h, linewidth=1,edgecolor=color,facecolor='none')
            ax.add_patch(rect)

            ax.text( (xmean*cell_w-w/2*img_w) / img_w, 1-(ymean*cell_h-hmean/2*img_h)/img_h-i*0.15, "%0.2f" % (confmean), transform=ax.transAxes, color=color)
            i+=1

plt.show()






