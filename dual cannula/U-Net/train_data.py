# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import my_alg

# define loss function
def ave_cross_en( y_true, y_pred):
    loss = ( keras.losses.kullback_leibler_divergence( y_true, y_pred) +  keras.losses.kld( 1 - y_true, 1- y_pred))
    return loss
    
def display_test_images(data_gen,test_gen, predict_gen, num=1000):

    os.mkdir('./UNetfigures' + '/bead')

    my_alg.generate_images_test(data_gen,test_gen, predict_gen, num=num, train_str='test')
    
# load dataset
mnist_x = np.load('/uufs/chpc.utah.edu/common/home/u1303742/TFModule/optrode/dataset/bead/refL_bead.npy')
mnist_b = np.load('/uufs/chpc.utah.edu/common/home/u1303742/TFModule/optrode/dataset/bead/ccmL_bead.npy' ) 


# expand dimension of dataset to fit model
mnist_b = np.expand_dims( mnist_b, axis=3)
mnist_x = np.expand_dims( mnist_x, axis=3)

mnist_x.shape, mnist_b.shape



# seprate dataset into train data, eval data and test data, randomly
NUM_TRAIN = int(  len( mnist_x))
x_train = mnist_x[1500:NUM_TRAIN]
b_train = mnist_b[1500:NUM_TRAIN]

x_eval = mnist_x[1000:1500]
b_eval = mnist_b[1000:1500]

x_test = mnist_x[0:1000]
b_test = mnist_b[0:1000]




fig = my_alg.show_random_sampels( b_train,10,1,idx = range(10))

# define model and compile it
model = my_alg.build_model_128()
model.compile(optimizer=keras.optimizers.Adam(1e-3, amsgrad= True), loss=ave_cross_en, metrics=['mae'])

# train model: set batch_size and epochs
model.fit(b_train, x_train, batch_size=16, epochs=30, verbose=1,validation_data= ( b_eval, x_eval))
model.save('/uufs/chpc.utah.edu/common/home/u1303742/TFModule/optrode/mymodel/left/BEAD')
model1 = tf.keras.models.load_model('/uufs/chpc.utah.edu/common/home/u1303742/TFModule/optrode/mymodel/left/BEAD', custom_objects={"ave_cross_en":ave_cross_en})

model1.summary()

train_pred = model.predict( b_train)
index_1 = np.array(range(len(b_train)))


# predict test data
test_pred = model.predict( b_test)
index = np.array(range(len(b_test)))

#
from skimage.metrics import structural_similarity as ssim
def average_SSIM( y_true, y_pred):
    ssims = []
    y_true = np.squeeze( y_true)
    y_pred = np.squeeze( y_pred)

    if len( y_true) !=  len( y_pred):
        raise  NameError("length not match!")
    for i in range(len(y_pred)):
        x = y_pred[i]
        y = y_true[i]
        ssims.append(ssim(x, y, gaussian_weights=True, data_range=255.0, k1=0.01, k2=0.03))
    ssims = np.array(ssims)
    ave_ssim = np.average(ssims)
    return ave_ssim
from sklearn.metrics import mean_absolute_error as me
def average_MAE( y_true, y_pred):
    mae = []
    y_true = np.squeeze( y_true)
    y_pred = np.squeeze( y_pred)
    if len( y_true) !=  len( y_pred):
        raise  NameError("length not match!")
    a,b = y_true[1].shape
    for i in range(len(y_pred)):
        x = y_pred[i]
        y = y_true[i]
        d = abs(x-y)
        m = sum(sum(d))/(a*b)
        mae.append(me(x, y))
    mae = np.array(mae)
    ave_mae = np.average(mae)
    return ave_mae
x_test_type = x_test.astype(np.float32)
res_SSIM = average_SSIM(x_test_type, test_pred)
print(res_SSIM)
res_mae = average_MAE(x_test, test_pred)
print(res_mae)

display_test_images(x_test,b_test,test_pred ,num=1000)