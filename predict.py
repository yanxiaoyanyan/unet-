from keras.layers import *
import scipy.misc
from vgg import model
from process import image_process
# from resnet101 import model

TEST = 'x_data/test/Image/'
IMG_HEIGHT, IMG_WIDTH = 512, 512
z_data = image_process(TEST, IMG_HEIGHT, IMG_WIDTH)
z_data = z_data[:, :, :, np.newaxis]

print('start predicting')
model.load_weights('lung_x_vgg.h5')
predict_image = model.predict(z_data, batch_size=2, verbose=1)
np.save('predict.npy', predict_image)

imgs_test = np.load('predict.npy')
print(imgs_test.shape)
for i in range(predict_image.shape[0]):
    B = imgs_test[i, :, :, 0]
    scipy.misc.imsave("result/im" + str(i) + "_testResults" + ".png", B)
