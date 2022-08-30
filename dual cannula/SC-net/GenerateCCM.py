import time
import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from Data import *
from Losses import *
from PatchGAN import *
from CycleReconstructor import *
from Unet import *

def get_and_increment_chpt_num(filename="chpt.txt"):
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val

PARAMS = {
    'change': 'Testing 200 image generation',

    'dir': 'glia', # Head directory inside datasets folder. Should contain ccm folder and ref folder if evaluating.
    'num_channels': 1, 'raw_input_shape': (128,128), 'target_input_shape': (128,128),

    'chpt': get_and_increment_chpt_num(),
    'channel_format': 'NCHW',

    'load_chpt': 150, # Presaved model checkpoint

    'evaluate': True,  # Set to false if no ground truth CCM images
    'num_images_to_generate': 10,
    'start_ds_index': 1, # CCM jpeg image start index, e.g. start_ds_index = 30400 with num_images_to_generate = 200 will use input ccm/30400.jpeg, ..., ccm/30599.jpeg

    'batch_size': 2,

    'G_PARAMS': {
        'filters': {'down':[64, 128, 256, 256], 'up':[256, 128, 64]},
        'dropout': {'down':[0.0, 0.0, 0.0, 0.0], 'up':[0.0, 0.0, 0.0]},
        'kernels': [5,5], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'instance', # 'instance', 'batch'
        'activation': 'relu',
    },
}

if PARAMS['channel_format'] == 'NCHW': tf.keras.backend.set_image_data_format('channels_first')
metrics = {'MAE': [], 'SSIM': [], 'PSNR': []}

class GenerateCCM:

    def log_parameters(self):
        print()
        print('PARAMETERS')
        for key,value in PARAMS.items(): print(key + ': ' + str(value))
        print("\n\n")
        self.G.summary()
        print("\n\n")
        print('PARAMETERS')
        for key,value in PARAMS.items(): print(key + ': ' + str(value))
        print("\n\n")

    def setup_model(self):
        self.G = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/G')
    
    def log_metrics(self):
        print('Evaluation:', end='')
        for key,metric in metrics.items(): 
            print(" {}: {:.4f}".format(key, np.average(metric)), end=',')
        print('\n', end='', flush=True)

    def generate_and_evaluate(self):
        self.setup_model()
        self.log_parameters()

        # Setup dataset
        ds = DataSetup(PARAMS['dir'], PARAMS['batch_size'], PARAMS['raw_input_shape'])
        data_generator = ds.setup_ccm_generation_dataset(PARAMS['num_images_to_generate'], PARAMS['evaluate'], PARAMS['start_ds_index'])
        
        os.mkdir('./figures/' + str(PARAMS['chpt']))

        i = 0
        for pair in data_generator:
            if PARAMS['evaluate']:
                x, y = pair
            else:
                x = pair
                y = None

            gx = self.G(x, training=True)
            generate_ccm_batch(PARAMS['chpt'], i, PARAMS['batch_size'], gx, x, y)
            
            # Evaluate
            if PARAMS['evaluate']:
                batch_metrics = self.evaluate(gx, y)
                for key,value in batch_metrics.items():
                    metrics[key].append(value)

            i += PARAMS['batch_size']
            print(i, flush=True)

        if PARAMS['evaluate']: self.log_metrics()
        print('Done', flush=True)
        
    def evaluate(self, gx, y):
        mae = reconstruction_loss(y, gx, dist_mae_loss)
        ssim, psnr = dist_compute_metrics(y, gx)
        return {'MAE': mae, 'SSIM': ssim, 'PSNR': psnr}


def main():
    generateCCM = GenerateCCM()
    generateCCM.generate_and_evaluate()


if __name__ == '__main__':
    main()