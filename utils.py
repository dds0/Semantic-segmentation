import os
import matplotlib.pyplot as plt
import config
from config import imshape, labels, hues
import cv2
import numpy as np
import tensorflow as tf

def sorted_dir(directory):
    return sorted(os.listdir(directory), key=lambda named: int(named.split('.')[0]))

def draw_mask_from_array(img, shapes, isBinary):
    if isBinary:
        quantity = 2
        classes = 1
    else:
        classes = config.n_classes
        quantity = int(len(config.hues)) + 2

    plt.figure(figsize=(quantity + 2,quantity + 2))
    plt.subplot(quantity, quantity, 1)
    img = tf.keras.preprocessing.image.array_to_img(img)
    plt.imshow(img)
    for _ in range(classes):
        plt.subplot(quantity, quantity, _ + 2)
        plt.imshow(shapes[:,:,_], cmap='gray')
    plt.show()

def add_masks(pred):
    blank = np.zeros(shape=imshape, dtype=np.uint8)

    for i, label in enumerate(labels):

        hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
        sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
        if config.isBinary:
            val = pred[:, :, 0].astype(np.uint8)
        else:
            val = pred[:,:,i].astype(np.uint8)

        im_hsv = cv2.merge([hue, sat, val])
        im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        blank = cv2.add(blank, im_rgb)

    for _ in range(config.n_classes):
        array = np.empty((256,256), dtype=np.uint8)

    return blank

def createRect(img): # shape[1] = x, shape[0] = y
    width = 2
    x = int(img.shape[1]/2 - config.sizeSquare/2 - width)
    y = int(img.shape[0]/2 - config.sizeSquare/2 - width)
    cv2.rectangle(img, (x, y), (x + config.sizeSquare + 2*width, y + config.sizeSquare + 2*width),
                  (255, 0, 0), width)
    return img

def save(img, dataset_location, i):
    x = int(img.shape[1]/2 - config.sizeSquare/2)
    y = int(img.shape[0]/2 - config.sizeSquare/2)
    arr = np.empty((imshape[0], imshape[1], imshape[2]), dtype=np.int32)
    arr = img[y:y + config.sizeSquare, x:x + config.sizeSquare,]

    dataset_location = os.path.join(dataset_location, str(i) + '.jpg')
    cv2.imwrite(dataset_location, arr)
    print(str(i) + '.jpg is saved.')
    return i + 1

def show(predic, img, isReturn = False, isReturnBinary = False):
    mask = predic.squeeze() * 255.0

    if config.isBinary:
        mask = np.expand_dims(mask, axis=2)

    mask = add_masks(mask)
    if isReturnBinary:
        return mask

    mask = np.array(mask, dtype=np.uint8)
    mask = cv2.addWeighted(img, 1.0, mask, 1.0, 0)
    if isReturn == 0:
        plt.imshow(mask)
        plt.show()

    if isReturn:
        return mask


def makePostProcessingVideo(img_dir, model, isReturnBinary = False):
    for _ in range(len(img_dir)):
        img, data = PreProcessingPhoto(img_dir[_])

        predict = model.predict(data)
        predict = show(predict, img, True, isReturnBinary)
        predict = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
        cv2.imwrite('predict/' + str(_) + '.jpg', predict)

def PreProcessingPhoto(img_dir):
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = tf.keras.preprocessing.image.img_to_array(img)
    data = np.expand_dims(data, axis=0)
    return img, data