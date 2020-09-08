import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.losses import binary_crossentropy
# from vgg import model
from process import image_process
from augument import my_generator
from keras.callbacks import TensorBoard
import time


# from resnet101 import model
# from vgg_plus import model
from resnet_plus import model

IMAGE_LIB = 'x_data/train/Image/'
MASK_LIB = 'x_data/train/Label/'
IMG_HEIGHT, IMG_WIDTH = 512, 512

x_data = image_process(IMAGE_LIB, IMG_HEIGHT, IMG_WIDTH)
y_data = image_process(MASK_LIB, IMG_HEIGHT, IMG_WIDTH)

x_data = x_data[:, :, :, np.newaxis]
y_data = y_data[:, :, :, np.newaxis]

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.25)

model_name = "predict-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

IMAGE_NUMBERS = len(x_train)
BATCH_SIZE = 1
EPOCHS = 50


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


def dice_p_bce(in_gt, in_pred, a=0.5, b=0.5):
    return a * binary_crossentropy(in_gt, in_pred) + b * dice_coef_loss(in_gt, in_pred)


model.compile(optimizer=Adam(1e-5), loss=dice_p_bce, metrics=['accuracy'])

weight_saver = ModelCheckpoint('lung_x_resnet++.h5',
                               monitor='val_acc',
                               save_best_only=True,
                               save_weights_only=True)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
print('start fitting')

hist = model.fit_generator(my_generator(x_train, y_train, BATCH_SIZE),
                           steps_per_epoch=IMAGE_NUMBERS/BATCH_SIZE,
                           validation_data=(x_val, y_val),
                           epochs=EPOCHS,
                           verbose=1,
                           callbacks=[tensorboard, weight_saver, annealer])

# model.save('lung_x_vgg.h5')
