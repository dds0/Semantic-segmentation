from config import imshape, n_classes, model_name
import tensorflow
from keras import backend as K

def jacard_coeff(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def normilize_img(x):
    x /= 255.0
    x -= 0.5
    x *= 2.
    return x

def unet(pretrained=False, base=4):

    if n_classes == 1:
        loss = 'binary_crossentropy'
        final_act = 'sigmoid'
    elif n_classes > 1:
        loss = 'categorical_crossentropy'
        final_act = 'softmax'

    b = base
    i = tensorflow.keras.layers.Input((imshape[0], imshape[1], imshape[2]))
    s = tensorflow.keras.layers.Lambda(lambda x: normilize_img(x))(i)

    c1 = tensorflow.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = tensorflow.keras.layers.Dropout(0.1) (c1)
    c1 = tensorflow.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = tensorflow.keras.layers.MaxPooling2D((2, 2)) (c1)

    c2 = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = tensorflow.keras.layers.Dropout(0.1) (c2)
    c2 = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = tensorflow.keras.layers.MaxPooling2D((2, 2)) (c2)

    c3 = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = tensorflow.keras.layers.Dropout(0.2) (c3)
    c3 = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = tensorflow.keras.layers.MaxPooling2D((2, 2)) (c3)

    c4 = tensorflow.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = tensorflow.keras.layers.Dropout(0.2) (c4)
    c4 = tensorflow.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = tensorflow.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = tensorflow.keras.layers.Dropout(0.3) (c5)
    c5 = tensorflow.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = tensorflow.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = tensorflow.keras.layers.concatenate([u6, c4])
    c6 = tensorflow.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = tensorflow.keras.layers.Dropout(0.2) (c6)
    c6 = tensorflow.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = tensorflow.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = tensorflow.keras.layers.concatenate([u7, c3])
    c7 = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = tensorflow.keras.layers.Dropout(0.2) (c7)
    c7 = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = tensorflow.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = tensorflow.keras.layers.concatenate([u8, c2])
    c8 = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = tensorflow.keras.layers.Dropout(0.1) (c8)
    c8 = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = tensorflow.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = tensorflow.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tensorflow.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = tensorflow.keras.layers.Dropout(0.1) (c9)
    c9 = tensorflow.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    o = tensorflow.keras.layers.Conv2D(n_classes, (1, 1), activation=final_act) (c9)

    model = tensorflow.keras.models.Model(inputs=i, outputs=o, name=model_name)
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(1e-4),
                  loss=loss,
                  metrics=[jacard_coeff])

    if pretrained == True:
        model.load_weights('hand.h5')

    model.summary()

    return model