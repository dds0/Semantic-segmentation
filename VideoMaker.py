import cv2
import numpy as np
import utils
import os
import model
import config

dataset_location = 'test'
dataset_path = utils.sorted_dir(dataset_location)

model = model.unet(True, config.n_classes)

if len(dataset_path) == 0:
    i = 0
else:
    i = int(dataset_path[len(dataset_path) - 1].split('.')[0]) + 1

isRun = True
capture = cv2.VideoCapture(0)
while isRun:
    ret, img = capture.read()
    img = utils.createRect(img)
    cv2.imshow('camera',img)
    k = cv2.waitKey(15) & 0xFF

    if k == 83:
        while True:
            ret, img = capture.read()
            img = utils.createRect(img)
            cv2.imshow('camera', img)
            k = cv2.waitKey(15) & 0xFF
            i = utils.save(img, dataset_location, i)
            print(i, 'save')
            if k == 81:
                isRun = False
                break
    if k == 81:
        isRun = False
    pass

capture.release()
cv2.destroyAllWindows()

in_path = [os.path.join('test', name) for name in utils.sorted_dir('test')]
utils.makePostProcessingVideo(in_path, model, isReturnBinary=True)
out = cv2.VideoWriter('output_video_binary.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (256,256))
out_path = [os.path.join('predict', name) for name in utils.sorted_dir('predict')]

for _ in range(len(out_path)):
    img = cv2.imread(out_path[_])
    out.write(img)



out.release()

