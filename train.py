import matplotlib.pyplot as plt
import numpy as np
import model
import cv2
import config
import tensorflow as tf
from PIL import Image
import data_generator
import utils
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if config.isHand:
    img_path = [os.path.join('dataset', name) for name in utils.sorted_dir('dataset')]
    ann_path = [os.path.join('dataset_masks', name) for name in utils.sorted_dir('dataset_masks')]

img_test = img_path[0:len(img_path)]
ann_test = ann_path[0:len(ann_path)]

if config.isHand:
    model = model.unet(True, config.n_classes)
else:
    model = model.unet(False, 1)

tg = data_generator.DataGenerator(img_paths=img_test, ann_paths=ann_test,
                   batch_size=12, isAugment=True, isShuffle=True, isBinary=config.isBinary)


#for _ in range(len(img_test)):
#    img, data = utils.PreProcessingPhoto(img_test[_])
#    predict = model.predict(data)
#    utils.show(predict, img, False)

#for _ in range(len(img_test)):
#    img, mask = tg.__getitem__(_)
#    utils.draw_mask_from_array(model.tensorflow.keras.preprocessing.image.array_to_img(img[0]), mask[0], config.isBinary)

while True:
    img, data = utils.PreProcessingPhoto(img_test[0])
    predict = model.predict(data)
    utils.show(predict, img, False)
    model.fit_generator(generator=tg, steps_per_epoch=len(tg),epochs=20,verbose=1)
    model.save_weights('hand.h5')


#for _ in range(len(img_path)):
#    img = cv2.imread(img_path[_])
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    masks = tg.create_binary_mask(256,tg.get_polygons(ann_path[_]))
#    utils.draw_mask_from_array(img, masks, True)

#for _ in range(len(img_path)):
#    img = cv2.imread(img_path[_])
#    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = data_generator.seq(image=img)
#    cv2.imshow("image",img)
#    cv2.waitKey(0)
