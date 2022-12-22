sizeSquare = 256
imshape = (256,256,3)

mode = 'binary'
model_name = 'u_net_' + mode
logbase = 'logs'

isHand = True

if isHand:
        hues = {'thumb': 90,#182
                'index finger': 90,#356
                'middle finger': 90,#31
                'ring finger': 90,#243
                'little finger': 90,#122
                'palm': 90}#90

labels = sorted(hues.keys())


if mode == 'multi':
    n_classes = len(labels) + 1
elif mode =='binary':
        n_classes = 1
        isBinary = True