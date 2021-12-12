import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import skimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from skimage.io import imsave as save


class DataSetup:
    def __init__(self, path, batch_size, size):
        self.data_path = path
        self.batch_size = batch_size
        self.image_size = size

    def __get_large_neuron_imgs(self):
        x, y = [], []
        for i in range(33493):
            x.append('./datasets/large-neuron/ref/' + str(i) + '.jpeg')
            y.append('./datasets/large-neuron/ccm/' + str(i) + '.jpeg')
        return x, y

    def __get_plant_imgs(self):
        x, y = [], []
        for i in range(11529):
            x.append('./datasets/plant-jpeg/ref/' + str(i) + '.jpeg')
            y.append('./datasets/plant-jpeg/ccm/' + str(i) + '.jpeg')
        return x, y

    def __get_4_layer_imgs(self):
        train_x, val_x, test_x, train_y, val_y, test_y = [],[],[],[],[],[]
        # Go through each directory, go through ccm then ref, set aside first 1000 into test and rest into train
        for dir in ['layer1','layer2','layer3','layer4']:
            for i in range(25): # 1000
                test_x.append('./datasets/cannula-4-layer/' + dir + '/ref-jpeg/' + str(i) + '.jpeg')
                test_y.append('./datasets/cannula-4-layer/' + dir + '/ccm-jpeg/' + str(i) + '.jpeg')
            for i in range(1000,2000):
                val_x.append('./datasets/cannula-4-layer/' + dir + '/ref-jpeg/' + str(i) + '.jpeg')
                val_y.append('./datasets/cannula-4-layer/' + dir + '/ccm-jpeg/' + str(i) + '.jpeg')
            
            num_files = len(os.listdir('./datasets/cannula-4-layer/' + dir + '/ref-jpeg/'))
            for i in range(2000, num_files):
                train_x.append('./datasets/cannula-4-layer/' + dir + '/ref-jpeg/' + str(i) + '.jpeg')
                train_y.append('./datasets/cannula-4-layer/' + dir + '/ccm-jpeg/' + str(i) + '.jpeg')

        # shuffle train and val
        p = np.random.permutation(len(train_x))
        train_x, train_y = np.array(train_x)[p], np.array(train_y)[p]
        p = np.random.permutation(len(val_x))
        val_x, val_y = np.array(val_x)[p], np.array(val_y)[p]
        test_x, test_y = np.array(test_x), np.array(test_y)

        return train_x, val_x, test_x, train_y, val_y, test_y

    def __get_brain_test_imgs(self):
        y = []
        for i in range(1255):
            y.append('./datasets/ccm-jpeg/' + str(i) + '.jpeg')
        return y

    def __get_thermal_vis_imgs(self):
        '''
        Gathers videos of 32 frames each, separates videos by 16 frames.
        '''
        # Gather videos of 32 frames, separated by 16 frames
        vid_files_vis_all, vid_files_therm_all = [], []
        vid_files_vis_curr, vid_files_therm_curr = [], []
        j = 0
        for i in range(62476):
            if j < 32:
                vid_files_vis_curr.append('../../thermal-vis/videos/thermal-128x128/' + str(i) + '.png')
                vid_files_therm_curr.append('../../thermal-vis/videos/vis-128x128/' + str(i) + '.png')
            elif j == 32 + 15:
                vid_files_vis_all.append(vid_files_vis_curr)
                vid_files_vis_curr = []
                vid_files_therm_all.append(vid_files_therm_curr)
                vid_files_therm_curr = []
                j = 0
                continue
            j += 1

        # shuffle train 
        p = np.random.permutation(len(vid_files_vis_all))
        vid_files_vis_all, vid_files_therm_all = np.array(vid_files_vis_all)[p], np.array(vid_files_therm_all)[p]

        # calculate number of train/val/test frames
        n_train = int(np.floor(len(vid_files_vis_all)*.8 * 32))
        n_val = int(np.floor(len(vid_files_vis_all)*.1 * 32))
        n_test = int(np.floor(len(vid_files_vis_all)*.1 * 32))

        # flatten videos into single array of frames
        vid_files_therm_all = np.ndarray.flatten(vid_files_therm_all)
        vid_files_vis_all = np.ndarray.flatten(vid_files_vis_all)

        print(vid_files_therm_all.shape, n_train)
        # separate into train/val/test
        train_x = vid_files_therm_all[:n_train]
        val_x = vid_files_therm_all[n_train:n_train + n_val]
        test_x = vid_files_therm_all[n_train + n_val:n_train + n_val + n_test]
        train_y = vid_files_vis_all[:n_train]
        val_y = vid_files_vis_all[n_train:n_train + n_val]
        test_y = vid_files_vis_all[n_train + n_val:n_train + n_val + n_test]

        return train_x, val_x, test_x, train_y, val_y, test_y
            
    def __get_bead_imgs(self):
        x, y = [], []
        for i in range(20000):
            x.append('./datasets/bead/ref/' + str(i) + '.jpeg')
            y.append('./datasets/bead/ccm/' + str(i) + '.jpeg')
        return x, y

    def setup_datasets(self):      
        shuffle = True 
        if self.data_path == 'cannula-neuron':
            x = np.load('./cannula-neuron/ref_imgs_128_1_12_2_3_modified1_50783_withlabel_20200501.npy', mmap_mode='r+')
            train_x = x[:45783,:,:,:]
            val_x = x[45783:48283,:,:,:] # 5% = 2500
            test_x = x[48283:,:,:,:] # 5% = 2500

            y = np.load('./cannula-neuron/ccm_imgs_128_1_12_2_3_modified1_50783_20200501.npy', mmap_mode='r+')
            train_y = y[:45783,:,:,:]
            val_y = y[45783:48283,:,:,:]
            test_y = y[48283:,:,:,:]

            preprocess = preprocess_neuron

        elif self.data_path == 'full-cannula-neuron':
            x, y = self.__get_large_neuron_imgs()
            train_x = x[:27400]
            val_x = x[27400:30400] # 5% = 2500
            test_x = x[30400:-1] # 5% = 2500

            train_y = y[:27400]
            val_y = y[27400:30400]
            test_y = y[30400:-1]

            preprocess = preprocess_large_neuron

        elif self.data_path == 'plant':
            x, y = self.__get_plant_imgs()
            train_x = x[:10368]
            val_x = x[10368:10944] # 5% = 576
            test_x = x[10944:11520] # 5% = 576

            train_y = y[:10368]
            val_y = y[10368:10944]
            test_y = y[10944:11520]

            preprocess = preprocess_large_neuron

        elif self.data_path == '4_layer':
            train_x, val_x, test_x, train_y, val_y, test_y = self.__get_4_layer_imgs()
            preprocess = preprocess_4_layer

        elif self.data_path == 'thermal-vis':
            train_x, val_x, test_x, train_y, val_y, test_y = self.__get_thermal_vis_imgs()
            preprocess = preprocess_thermal_vis
            shuffle = False

        elif self.data_path == 'bead' or self.data_path == 'bead2':
            x, y = self.__get_bead_imgs()
            train_x = x[:18000]
            val_x = x[18000:19008] 
            test_x = x[19008:] 

            train_y = y[:18000]
            val_y = y[18000:19008]
            test_y = y[19008:]

            preprocess = preprocess_bead

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        if shuffle:
            SHUFFLE_BUFFER_SIZE = 1600
            train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(SHUFFLE_BUFFER_SIZE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        else:
            train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return train_ds, val_ds, test_ds


    # Brain evaluation
    def setup_brain_eval_set(self):
        y = self.__get_brain_test_imgs()
        test_ds = tf.data.Dataset.from_tensor_slices((y))
        test_ds = test_ds.map(preprocess_brain_eval, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return test_ds


    # Generate CCM
    def __get_large_neuron_imgs_from_path(self, num_images_to_generate, evaluate, start_ds_index):
        x, y = [], []
        for i in range(start_ds_index, start_ds_index + num_images_to_generate):
            x.append('./datasets/' + self.data_path + '/ref/' + str(i) + '.jpeg')
            if evaluate:
                y.append('./datasets/' + self.data_path + '/ccm/' + str(i) + '.jpeg')

        if evaluate:
            return x, y
        else:
            return x, None

    def setup_ccm_generation_dataset(self, num_images_to_generate, evaluate, start_ds_index):
        x, y = self.__get_large_neuron_imgs_from_path(num_images_to_generate, evaluate, start_ds_index)
        if evaluate:
            preprocess = preprocess_large_neuron
            ds = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            preprocess = preprocess_large_neuron_ccm_only
            ds = tf.data.Dataset.from_tensor_slices((x))

        return ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)



def preprocess_neuron(x,y):
    y = tf.cast(y, tf.float32)
    y = normalize_max(y)

    x = tf.cast(x, tf.float32)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    #####
    return x, y

def preprocess_large_neuron(x,y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=1)
    y = tf.cast(y, tf.float32)
    y = normalize_max(y)

    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=1)
    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    #####
    return x, y

def preprocess_large_neuron_ccm_only(x):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=1)
    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    #####
    return x

def preprocess_4_layer(x,y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=1)
    y = tf.cast(y, tf.float32)
    y = tf.image.resize(y, (128,128))
    y = normalize_max(y)

    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=1)
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, (128,128))
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    #####
    return x, y

def preprocess_brain_eval(y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=1)
    y = tf.cast(y, tf.float32)
    y = tf.image.resize(y, (128,128))
    y = normalize_max(y)

    ##### NCHW
    y = tf.transpose(y, [2, 0, 1])
    #####
    return y

def preprocess_thermal_vis(x,y):
    y = tf.io.read_file(y)
    y = tf.image.decode_png(y, channels=3)
    y = tf.cast(y, tf.float32)
    y = normalize_255(y)

    x = tf.io.read_file(x)
    x = tf.image.decode_png(x, channels=3)
    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    #####
    return x, y

def preprocess_bead(x,y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=1)
    y = tf.cast(y, tf.float32)
    y = tf.image.resize(y, (128,128))
    y = normalize_max(y)

    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=1)
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, (128,128))
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    #####
    return x, y

def normalize_max(x):
    return x / tf.reduce_max(x)

def normalize_255(x):
    return x / 255.


def generate_images(G, F, data_gen, chpt, epoch, batch_size, num=1, num_channels=3, train_str='train'):
    displayed = 0
    for x, y in data_gen:
        f_prediction = F(y, training=True)
        g_prediction = G(x, training=True)
        f_g_x_prediction = F(f_prediction, training=True)
        g_f_y_prediction = G(g_prediction, training=True)

        ##### NCHW
        x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.transpose(y, [0, 2, 3, 1])
        f_prediction = tf.transpose(f_prediction, [0, 2, 3, 1])
        g_prediction = tf.transpose(g_prediction, [0, 2, 3, 1])
        f_g_x_prediction = tf.transpose(f_g_x_prediction, [0, 2, 3, 1])
        g_f_y_prediction = tf.transpose(g_f_y_prediction, [0, 2, 3, 1])
        #####

        for j in range(batch_size):
            plt.figure(figsize=(12, 12))

            display_list = [x[j], y[j], f_prediction[j], g_prediction[j], f_g_x_prediction[j], g_f_y_prediction[j]]
            title = ['Ground Truth', 'Input Image', 'F Predicted Image', 'G Predicted Image', 'F Cycled Image', 'G Cycled Image']
            fig_title = ['x', 'y', 'f_y', 'g_x', 'f_g_x', 'g_f_y']

            for i in range(6):
                plt.subplot(1, 6, i+1)
                plt.title(title[i])
                if num_channels == 1:
                    img = np.squeeze(display_list[i])
                    imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', img)
                else:
                    img = display_list[i]
                    save('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                plt.imshow(img)
                plt.axis('off')
            plt.savefig('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + train_str + '-' + str(displayed), bbox_inches='tight', pad_inches=0)
            displayed += 1
            if displayed == num: return


def generate_images_single_gen(F, data_gen, chpt, epoch, batch_size, num=1, num_channels=3, train_str='train'):
    displayed = 0
    for x, y in data_gen:
        f_prediction = F(y, training=True)

        ##### NCHW
        x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.transpose(y, [0, 2, 3, 1])
        f_prediction = tf.transpose(f_prediction, [0, 2, 3, 1])
        #####

        for j in range(batch_size):
            plt.figure(figsize=(12, 12))

            display_list = [x[j], y[j], f_prediction[j]]
            title = ['Ground Truth', 'Input Image', 'F Predicted Image']
            fig_title = ['x', 'y', 'f_y']

            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                if num_channels == 1:
                    img = np.squeeze(display_list[i])
                    imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', img)
                else:
                    img = display_list[i]
                    skimage.io.imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                plt.imshow(img)
                plt.axis('off')
            plt.savefig('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + train_str + '-' + str(displayed), bbox_inches='tight', pad_inches=0)
            displayed += 1
            if displayed == num: return


def generate_ccm_batch(chpt, start_idx, batch_size, pred, x, y=None):
    displayed = 0

    ##### NCHW
    if y is not None: y = tf.transpose(y, [0, 2, 3, 1])
    x = tf.transpose(x, [0, 2, 3, 1])
    pred = tf.transpose(pred, [0, 2, 3, 1])
    #####

    for j in range(batch_size):
        plt.figure(figsize=(12, 12))

        display_list = [x[j], pred[j]]
        title = ['Input', 'Output']
        if y is not None:
            display_list.append(y[j])
            title.append('Ground Truth')

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            img = np.squeeze(display_list[i])
            if i == 1: imsave('./figures/' + str(chpt) + '/' + str(displayed + start_idx) + '-CCM_Prediction.png', img)
            plt.imshow(img)
            plt.axis('off')
        plt.savefig('./figures/' + str(chpt) + '/' + str(displayed + start_idx), bbox_inches='tight', pad_inches=0)
        displayed += 1


def generate_images_unet(F, data_gen, chpt, epoch, batch_size, num=1, num_channels=3, train_str='train'):
    displayed = 0
    for x, y in data_gen:
        f_prediction = F(y, training=False)

        ##### NCHW
        x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.transpose(y, [0, 2, 3, 1])
        f_prediction = tf.transpose(f_prediction, [0, 2, 3, 1])
        #####

        for j in range(batch_size):
            plt.figure(figsize=(12, 12))

            display_list = [x[j], y[j], f_prediction[j]]
            title = ['Ground Truth', 'Input Image', 'F Predicted Image']
            fig_title = ['x', 'y', 'f_y']

            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                if num_channels == 1:
                    img = np.squeeze(display_list[i])
                    imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', img)
                else:
                    img = display_list[i]
                    skimage.io.imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                plt.imshow(img) # * 0.5 + 0.5) # tanh loss
                plt.axis('off')
            plt.savefig('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + train_str + '-' + str(displayed), bbox_inches='tight', pad_inches=0)
            displayed += 1
            if displayed == num: return


def generate_images_brain(F, data_gen, chpt, batch_size, num=1, num_channels=3, train_str='train'):
    displayed = 0
    for y in data_gen:
        f_prediction = F(y, training=False)

        ##### NCHW
        y = tf.transpose(y, [0, 2, 3, 1])
        f_prediction = tf.transpose(f_prediction, [0, 2, 3, 1])
        #####

        for j in range(batch_size):
            plt.figure(figsize=(12, 12))

            display_list = [y[j], f_prediction[j]]
            title = ['Input Image', 'F Predicted Image']
            fig_title = ['y', 'f_y']

            for i in range(2):
                plt.subplot(1, 2, i+1)
                plt.title(title[i])
                if num_channels == 1:
                    img = np.squeeze(display_list[i])
                    skimage.io.imsave('./figures/' + str(chpt) + '/brain/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', img)
                else:
                    img = display_list[i]
                    skimage.io.imsave('./figures/' + str(chpt) + '/brain/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                plt.imshow(img)
                plt.axis('off')
            plt.savefig('./figures/' + str(chpt) + '/brain-' + train_str + '-' + str(displayed), bbox_inches='tight', pad_inches=0)
            displayed += 1
            if displayed == num: return

def plot(vals, names, title, y_label, chpt, epoch):
    plt.rcParams.update({'font.family': 'serif'})
    plt.figure()
    for val in vals:
        plt.plot(val)
    plt.title(title)
    plt.legend(names)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.savefig('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + title + '.png')
    plt.show()