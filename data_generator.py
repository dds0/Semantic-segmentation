import matplotlib.pyplot as plt
import numpy as np
from config import imshape, n_classes, labels, model_name
import imgaug as ia
from imgaug.augmentables.polys import Polygon
import cv2
import json
import model
import utils

seq = ia.augmenters.Sequential([
    ia.augmenters.Fliplr(0.5),
    ia.augmenters.Flipud(0.5),
    ia.augmenters.Sometimes(0.25,
        ia.augmenters.Multiply((0.5, 1.5))
    ),
    ia.augmenters.Affine(
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-90,90),
    ),
    ia.augmenters.contrast.LinearContrast((0.8,1.2)),
    ia.augmenters.Sometimes(0.25,
        ia.augmenters.GaussianBlur(sigma=(0, 8))
    )
], random_order=True)

class DataGenerator(utils.tf.keras.utils.Sequence):
    def __len__(self):
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_paths))
        if self.isShuffle == True:
            np.random.shuffle(self.indexes)

    def get_polygons(self, ann_paths):
        with open(ann_paths) as jsonFile:
            data = json.load(jsonFile)

        polygons = []
        for shape in data['shapes']:
            polygons.append(Polygon(shape['points'], label=shape['label']))

        return polygons

    def create_multi_masks(self, sizeXY, polygons):

        label_with_polygons = {}
        for _ in polygons:
            label_with_polygons.update({_.label: np.array(_.coords, dtype=np.int32)})

        channels = []
        background = np.zeros(shape=(sizeXY, sizeXY), dtype=np.float32)
        background.fill(255)

        for i, cur_label in enumerate(labels):

            blank = np.zeros(shape=(sizeXY, sizeXY), dtype=np.float32)
            if cur_label in label_with_polygons.keys():
                cv2.fillPoly(blank, [label_with_polygons[cur_label]], 255)
                cv2.fillPoly(background, [label_with_polygons[cur_label]], 0)

            channels.append(blank)
        channels.append(background)

        multi_mask = np.stack(channels, axis=2) / 255.0
        return multi_mask

    def create_binary_mask(self, sizeXY, polygons):
        label_with_polygons = {}
        for _ in polygons:
            label_with_polygons.update({_.label: np.array(_.coords, dtype=np.int32)})

        mask = np.zeros(shape=(sizeXY, sizeXY), dtype=np.float32)

        for i, cur_label in enumerate(labels):
            if cur_label in label_with_polygons.keys():
                cv2.fillPoly(mask, [label_with_polygons[cur_label]], 255)

        mask = np.expand_dims(mask, axis=2)/ 255.0
        return mask

    def augment_poly(self, img, polygons):
        return seq(image=img, polygons=polygons)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        cur_img = [self.img_paths[k] for k in indexes]
        cur_ann = [self.ann_paths[k] for k in indexes]

        img_batch, mask_batch = self.__data_generation(cur_img, cur_ann)

        return img_batch, mask_batch

    def __data_generation(self, img_folder, ann_folder):

        images = np.empty((self.batch_size, imshape[0], imshape[1], imshape[2]), dtype=np.float32)
        masks = np.empty((self.batch_size, imshape[0], imshape[1], n_classes),  dtype=np.float32)

        for i, (curr_img, curr_ann) in enumerate(zip(img_folder, ann_folder)):
            img = cv2.imread(curr_img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            curr_polygons = self.get_polygons(curr_ann)

            if self.isAugment:
                img, curr_polygons = self.augment_poly(img, curr_polygons)

            if self.isBinary:
                mask = self.create_binary_mask(256, curr_polygons)
            else:
                mask = self.create_multi_masks(256, curr_polygons)

            img = model.tensorflow.keras.preprocessing.image.img_to_array(img)
            images[i,] = img
            masks[i,] = mask

        return images, masks

    def __init__(self, img_paths, ann_paths, batch_size=5,
                 isShuffle=True, isAugment=False, isBinary=False):
        self.img_paths = img_paths
        self.ann_paths = ann_paths
        self.batch_size = batch_size
        self.isShuffle = isShuffle
        self.isAugment = isAugment
        self.isBinary = isBinary
        self.on_epoch_end()


