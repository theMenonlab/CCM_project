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
from SRGAN import *


PARAMS = {
    'change': '',

    # Uncomment for dataset
    # 'dir': 'cannula-neuron', 'num_channels': 1, 'raw_input_shape': (128,128), 'target_input_shape': (128,128),
    # 'dir': 'plant', 'num_channels': 1, 'raw_input_shape': (468,468), 'target_input_shape': (468,468),
    # 'dir': 'full-cannula-neuron', 'num_channels': 1, 'raw_input_shape': (338,338), 'target_input_shape': (338,338),
    # 'dir': '4_layer', 'num_channels': 1, 'raw_input_shape': (447,447), 'target_input_shape': (447,447),
    #'dir': 'glia', 'num_channels': 1, 'raw_input_shape': (128,128), 'target_input_shape': (128,128),
    # 'dir': 'thermal-vis', 'num_channels': 3, 'raw_input_shape': (128,128), 'target_input_shape': (128,128),
     'dir': 'bead', 'num_channels': 1, 'raw_input_shape': (128,128), 'target_input_shape': (128,128),

    'generator': cycle_reconstructor, 
    'rec_loss': dist_mae_loss, # dist_mae_loss, dist_mse_loss,
    'cycle_loss': dist_mae_loss, # dist_mae_loss, dist_mse_loss,
    'gan_gen_loss': dist_ls_gan_loss_gen, # dist_ls_gan_loss_gen, dist_gan_loss_gen
    'gan_disc_loss': dist_ls_gan_loss_disc, # dist_ls_gan_loss_disc, dist_gan_loss_disc


    'chpt': 105,
    'channel_format': 'NCHW',
    'load_chpt': None, # Checkpoint number of model you would like to load or None for new model
    'save_models': True, # Saves model after testing

    'epochs': 150,
    'batch_size': 4,
    'learning_rate': 2e-4,
    'disc_lr': 2e-5,

    # Loss scalars:
        # type double: Loss = GAN + beta(MAE + lambda(Forward + lambda_b*Backward))
        # type double_no_disc: Loss = MAE + lambda(Forward + lambda_b*Backward)
    'lambda': 1,
    'lambda_b': 0.1,
    'beta': 100,

    'generator_to_discriminator_steps': 0, # 0: update both each step, 1: 1 generator step then 1 discriminator, 2: 2 generator steps then 1 discriminator, ...

    # Uncomment for training method
    'type': 'double', # Self Consistent Supervised
    # 'type': 'double_no_disc', # Self Consistent Supervised without GAN loss
    # 'type': 'single_no_disc', # Supervised only F, current model
    # 'type': 'unet', # Supervised only F, standard unet
     #'type': 'srgan', # Supervised only F, SRGAN architecture


    'F_PARAMS': {
        'filters': {'down': [64, 128, 256, 256], 'up': [256, 128, 64]},
        'dropout': {'down': [0.0, 0.0, 0.0, 0.0], 'up': [0.0, 0.0, 0.0]},
        'kernels': [3,3], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'instance', # 'instance', 'batch'
        'activation': 'relu',
        'skip': False,
    },
    'G_PARAMS': {
        'filters': {'down':[64, 128, 256, 256], 'up':[256, 128, 64]},
        'dropout': {'down':[0.0, 0.0, 0.0, 0.0], 'up':[0.0, 0.0, 0.0]},
        'kernels': [3,3], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'instance', # 'instance', 'batch'
        'activation': 'relu',
        'skip': False,
    },
}

# Setup additional model args based on PARAMS
if PARAMS['channel_format'] == 'NCHW': tf.keras.backend.set_image_data_format('channels_first')
PARAMS['F_PARAMS']['num_channels'] = PARAMS['G_PARAMS']['num_channels'] = PARAMS['num_channels']
PARAMS['F_PARAMS']['channel_format'] = PARAMS['G_PARAMS']['channel_format'] = PARAMS['channel_format']
PARAMS['F_PARAMS']['input_shape'] = PARAMS['G_PARAMS']['output_shape'] =  PARAMS['raw_input_shape']
PARAMS['F_PARAMS']['output_shape'] = PARAMS['G_PARAMS']['input_shape'] =  PARAMS['target_input_shape']

# Metric / Loss Dictionaries
reset_epoch_losses = {'dx_loss': [], 'dy_loss': [], 'f_loss': [], 'f_rec_loss': [], 'f_cycle': [], 'f_ssim': [], 'f_psnr': [], 'g_loss': [], 'g_rec_loss': [], 'g_cycle': []}
val_reset_epoch_losses = {'val_dx_loss': [], 'val_dy_loss': [], 'val_f_loss': [], 'val_f_rec_loss': [], 'val_f_cycle': [], 'val_f_ssim': [], 'val_f_psnr': [], 
                        'val_g_loss': [], 'val_g_rec_loss': [], 'val_g_cycle': []}
total_losses = {'dx_loss': [], 'dy_loss': [], 'f_loss': [], 'f_rec_loss': [], 'f_cycle': [], 'f_ssim': [], 'f_psnr': [], 'g_loss': [], 'g_rec_loss': [], 'g_cycle': [], 
                'val_dx_loss': [], 'val_dy_loss': [], 'val_f_loss': [], 'val_f_rec_loss': [], 'val_f_cycle': [], 'val_f_ssim': [], 'val_f_psnr': [], 
                'val_g_loss': [],'val_g_rec_loss': [], 'val_g_cycle': []}


class Train:

    def log_parameters(self):
        print()
        print('PARAMETERS')
        for key,value in PARAMS.items(): print(key + ': ' + str(value))
        print("\n\n")
        self.F.summary()
        if PARAMS['type'] == 'double': 
            print()
            self.Dx.summary()
            print()
            self.Dy.summary()
        elif PARAMS['type'] == 'srgan':
            print()
            self.Dx.summary()
        if (PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double') and PARAMS['raw_input_shape'] != PARAMS['target_input_shape']: 
            print() 
            self.G.summary()
        print("\n\n")
        print('PARAMETERS')
        for key,value in PARAMS.items(): print(key + ': ' + str(value))
        print("\n\n")

    def get_train_val_funcs(self):
        if PARAMS['type'] == 'double': 
            self.train_step = self.train_two_gen_step
            self.val_step = self.validate_two_gen_step
        elif PARAMS['type'] == 'double_no_disc': 
            self.train_step = self.train_two_gen_no_disc_step
            self.val_step = self.validate_two_gen_no_disc_step
        elif PARAMS['type'] == 'srgan': 
            self.train_step = self.train_unet_step
            self.val_step = self.validate_unet_step
        elif PARAMS['type'] == 'single_no_disc': 
            self.train_step = self.train_one_gen_no_disc_step
            self.val_step = self.validate_one_gen_no_disc_step
        elif PARAMS['type'] == 'unet':
            self.train_step = self.train_unet_step
            self.val_step = self.validate_unet_step

    def setup_models_and_optimizers(self):
        self.b = PARAMS['beta']
        self.l = PARAMS['lambda']
        self.lb = PARAMS['lambda_b']

        if PARAMS['type'] == 'unet':
            if PARAMS['load_chpt']:
                self.F = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/F')
            else:
                self.F = Unet(input_shape=PARAMS['F_PARAMS']['input_shape'], num_channels=PARAMS['F_PARAMS']['num_channels'])
            self.f_optimizer = tf.keras.optimizers.Adam(PARAMS['learning_rate'])
            self.f_optimizer._create_all_weights(self.F.trainable_weights)

        elif PARAMS['type'] == 'srgan':
            if PARAMS['load_chpt']:
                self.F = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/F')
                self.Dx = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/Dx')
            else:
                self.F = srgan_gen(input_shape=PARAMS['F_PARAMS']['input_shape'], channel_format=PARAMS['F_PARAMS']['channel_format'], num_channels=PARAMS['F_PARAMS']['num_channels'], num_residuals=8)
                self.Dx = srgan_disc(PARAMS['F_PARAMS']['output_shape'], PARAMS['F_PARAMS']['channel_format'], PARAMS['F_PARAMS']['num_channels'])
            self.f_optimizer = tf.keras.optimizers.Adam(PARAMS['learning_rate'])
            self.f_optimizer._create_all_weights(self.F.trainable_weights)
            self.dx_optimizer = tf.keras.optimizers.Adam(PARAMS['disc_lr'], beta_1=0.5)
            self.dx_optimizer._create_all_weights(self.Dx.trainable_weights)

        else:
            if PARAMS['load_chpt']:
                self.F = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/F')
            else:
                self.F = PARAMS['generator'](PARAMS['F_PARAMS'])
            self.f_optimizer = tf.keras.optimizers.Adam(PARAMS['learning_rate'], beta_1=0.5)
            self.f_optimizer._create_all_weights(self.F.trainable_weights)

        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            if PARAMS['load_chpt']:
                self.G = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/G')
            else:
                self.G = PARAMS['generator'](PARAMS['G_PARAMS'])
            self.g_optimizer = tf.keras.optimizers.Adam(PARAMS['learning_rate'], beta_1=0.5)
            self.g_optimizer._create_all_weights(self.G.trainable_weights)
                
        if PARAMS['type'] == 'double':
            if PARAMS['load_chpt']:
                self.Dx = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/Dx')
                self.Dy = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/Dy')
            else:
                self.Dx = patch_gan(PARAMS['F_PARAMS']['output_shape'], PARAMS['F_PARAMS']['num_channels'], PARAMS['F_PARAMS']['norm'])
                self.Dy = patch_gan(PARAMS['G_PARAMS']['output_shape'], PARAMS['G_PARAMS']['num_channels'], PARAMS['G_PARAMS']['norm'])
            self.dx_optimizer = tf.keras.optimizers.Adam(PARAMS['disc_lr'], beta_1=0.5)
            self.dy_optimizer = tf.keras.optimizers.Adam(PARAMS['disc_lr'], beta_1=0.5)
            self.dx_optimizer._create_all_weights(self.Dx.trainable_weights)
            self.dy_optimizer._create_all_weights(self.Dy.trainable_weights)

    def load_old_optimizers_if_necessary(self):
        if PARAMS['load_chpt'] is None: return
        self.load_optimizer_state(self.f_optimizer, 'F', self.F)
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            self.load_optimizer_state(self.g_optimizer, 'G', self.G)
        if PARAMS['type'] == 'double':
            self.load_optimizer_state(self.dx_optimizer, 'Dx', self.Dx)
            self.load_optimizer_state(self.dy_optimizer, 'Dy', self.Dy)
        if PARAMS['type'] == 'srgan':
            self.load_optimizer_state(self.dx_optimizer, 'Dx', self.Dx)

    def save_models(self):
        os.mkdir('./models/' + str(PARAMS['chpt']))
        self.F.save('./models/' + str(PARAMS['chpt']) + '/F')
        self.save_optimizer_state(self.f_optimizer, 'F')
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            self.G.save('./models/' + str(PARAMS['chpt']) + '/G')
            self.save_optimizer_state(self.g_optimizer, 'G')
        if PARAMS['type'] == 'double':
            self.Dx.save('./models/' + str(PARAMS['chpt']) + '/Dx')
            self.save_optimizer_state(self.dx_optimizer, 'Dx')
            self.Dy.save('./models/' + str(PARAMS['chpt']) + '/Dy')
            self.save_optimizer_state(self.dy_optimizer, 'Dy')
        elif PARAMS['type'] == 'srgan':
            self.Dx.save('./models/' + str(PARAMS['chpt']) + '/Dx')
            self.save_optimizer_state(self.dx_optimizer, 'Dx')

    # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state/49504376
    def save_optimizer_state(self, optimizer, opt_name):
        np.save('./models/' + str(PARAMS['chpt']) + '/' + opt_name + '-opt', optimizer.get_weights())

    def load_optimizer_state(self, optimizer, opt_name, model):
        '''
        Loads keras.optimizers object state.

        Arguments:
        optimizer --- Optimizer object to be loaded.
        load_path --- Path to save location.
        load_name --- Name of the .npy file to be read.
        model_train_vars --- List of model variables (obtained using Model.trainable_variables)
        '''
        # # Load optimizer weights
        opt_weights = np.load('./models/' + str(PARAMS['load_chpt']) + '/' + opt_name + '-opt.npy', allow_pickle=True)
        # Set the weights of the optimizer
        optimizer.set_weights(opt_weights)

    def display_images(self, epoch, train_gen, val_gen):
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            generate_images(self.G, self.F, train_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=5, num_channels=PARAMS['num_channels'], train_str='train')
            val_num = 20 if epoch > 20 else 5
            generate_images(self.G, self.F, val_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=val_num, num_channels=PARAMS['num_channels'], train_str='val')
        elif PARAMS['type'] == 'single_no_disc' or PARAMS['type'] == 'unet' or PARAMS['type'] == 'srgan':
            if PARAMS['type'] == 'single_no_disc': gen_images = generate_images_single_gen
            else: gen_images = generate_images_unet
            os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            gen_images(self.F, train_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=10, num_channels=PARAMS['num_channels'], train_str='train')
            gen_images(self.F, val_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=10, num_channels=PARAMS['num_channels'], train_str='val')

    def display_test_images(self, epoch, test_gen, num=100, ccm_only=False, pfa=False):
        if ccm_only:
            if not pfa:            	
                os.mkdir('./figures/' + str(PARAMS['chpt']) + '/brain')
                generate_images_brain(self.F, test_gen, PARAMS['chpt'], PARAMS['batch_size'], num=num, num_channels=PARAMS['num_channels'], train_str='test')
            else:
                os.mkdir('./figures/' + str(PARAMS['chpt']) + '/pfa_brain')
                generate_images_brain(self.F, test_gen, PARAMS['chpt'], PARAMS['batch_size'], num=num, num_channels=PARAMS['num_channels'], train_str='test', pfa=True)
            
        elif (PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double'):
            os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            generate_images(self.G, self.F, test_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=num, num_channels=PARAMS['num_channels'], train_str='test')
        elif PARAMS['type'] == 'single_no_disc' or PARAMS['type'] == 'unet' or PARAMS['type'] == 'srgan':
            if PARAMS['type'] == 'single_no_disc': gen_images = generate_images_single_gen
            else: gen_images = generate_images_unet
            os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            gen_images(self.F, test_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=num, num_channels=PARAMS['num_channels'], train_str='test')

    def plot(self, epoch):
        if PARAMS['type'] == 'single_no_disc':
            vals = [total_losses['f_loss'], total_losses['val_f_loss']]
            names = ['Train', 'Validation']
            plot(vals, names, 'MAE', 'MAE', PARAMS['chpt'], epoch)
        elif PARAMS['type'] == 'double_no_disc':
            vals = [total_losses['f_loss'], total_losses['val_f_loss'], total_losses['g_loss'], total_losses['val_g_loss']]
            names = ['F Train', 'F Validation', 'G Train', 'G Validation']
            plot(vals, names, 'Losses', 'Loss', PARAMS['chpt'], epoch)
        elif PARAMS['type'] == 'double':
            vals = [total_losses['f_loss'], total_losses['g_loss']]
            names = ['F', 'G']
            plot(vals, names, 'Generator Losses', 'Loss', PARAMS['chpt'], epoch)
            vals = [total_losses['dx_loss'], total_losses['dy_loss']]
            names = ['Dx', 'Dy']
            plot(vals, names, 'Discriminator Losses', 'Loss', PARAMS['chpt'], epoch)
    
    def log_epoch_stats(self, epoch, start, batch_losses, val_losses, epoch_losses, epoch_val_losses):
        if epoch is not None:
            print('E{}, {}s:'.format(epoch, int(time.time()-start)), end='')
        if batch_losses is not None:
            for key,_ in batch_losses.items(): 
                print(" {}: {:.4f}".format(key, np.average(epoch_losses[key])), end=',')
                total_losses[key].append(np.average(epoch_losses[key]))
        if val_losses is not None:
            for key,_ in val_losses.items(): 
                print(" {}: {:.3f}".format(key, np.average(epoch_val_losses[key])), end=',')
                total_losses[key].append(np.average(epoch_val_losses[key]))
        print('\n', end='', flush=True)


    def train(self):
        # Setup multi-gpu 
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with self.mirrored_strategy.scope():
            self.setup_models_and_optimizers()
        self.log_parameters()

        # Setup datasets
        ds = DataSetup(PARAMS['dir'], PARAMS['batch_size'], PARAMS['raw_input_shape'])
        train_gen, val_gen, test_gen = ds.setup_datasets()
        dist_train_gen = self.mirrored_strategy.experimental_distribute_dataset(train_gen)
        dist_val_gen = self.mirrored_strategy.experimental_distribute_dataset(val_gen)
        dist_test_gen = self.mirrored_strategy.experimental_distribute_dataset(test_gen)

        # Setup training/validation functions
        self.get_train_val_funcs()

        os.mkdir('./figures/' + str(PARAMS['chpt']))

        for epoch in range(1, PARAMS['epochs'] + 1):
            start = time.time()
            # Train Loop
            step = 0
            epoch_losses = dict(reset_epoch_losses)
            for x, y in dist_train_gen:
                if PARAMS['generator_to_discriminator_steps'] == 0:
                    batch_losses = self.distributed_train_step(x, y)[0]
                else:
                    if step % (PARAMS['generator_to_discriminator_steps'] + 1) == 0:
                        batch_losses = self.distributed_train_disc_step(x, y)[0]
                    else:
                        batch_losses = self.distributed_train_gen_step(x, y)[0]
                step += 1
                
                for key,value in batch_losses.items():
                    epoch_losses[key].append(self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None))
                
            # Validation Loop
            epoch_val_losses = dict(val_reset_epoch_losses)
            for x, y in dist_val_gen:
                val_losses = self.distributed_val_step(x, y)[0]
                for key,value in val_losses.items(): 
                    epoch_val_losses[key].append(self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None))

            self.log_epoch_stats(epoch, start, batch_losses, val_losses, epoch_losses, epoch_val_losses)
            if epoch % 5 == 0: 
                self.display_images(epoch, train_gen, val_gen)
                self.plot(epoch)
        
        # Test loop
        test_losses = dict(val_reset_epoch_losses)
        for x, y in dist_test_gen:
            step_losses = self.distributed_val_step(x, y)[0]
            for key,value in step_losses.items(): 
                test_losses[key].append(self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None))

        self.log_epoch_stats(None, None, None, step_losses, None, test_losses)
        self.display_test_images(0, test_gen)
        
        if PARAMS['save_models']:
            self.save_models()

        # Test on brain samples
        brain_gen = ds.setup_brain_eval_set()
        self.display_test_images(-2, brain_gen, num=300, ccm_only=True)

        pfa_brain_gen = ds.setup_pfa_brain_eval_set()
        self.display_test_images(-2, pfa_brain_gen, num=170, ccm_only=True, pfa=True)

    @tf.function
    def distributed_train_gen_step(self, x, y):
        per_replica_losses = self.mirrored_strategy.run(self.train_two_gen_gen_step, args=(x,y))
        replica_losses = self.mirrored_strategy.experimental_local_results(per_replica_losses)
        return replica_losses

    @tf.function
    def distributed_train_disc_step(self, x, y):
        per_replica_losses = self.mirrored_strategy.run(self.train_two_gen_disc_step, args=(x,y))
        replica_losses = self.mirrored_strategy.experimental_local_results(per_replica_losses)
        return replica_losses

    @tf.function
    def distributed_train_step(self, x, y):
        per_replica_losses = self.mirrored_strategy.run(self.train_step, args=(x,y))
        replica_losses = self.mirrored_strategy.experimental_local_results(per_replica_losses)
        return replica_losses

    @tf.function
    def distributed_val_step(self, x, y):
        per_replica_losses = self.mirrored_strategy.run(self.val_step, args=(x,y))
        replica_losses = self.mirrored_strategy.experimental_local_results(per_replica_losses)
        return replica_losses
    

    @tf.function
    def train_two_gen_gen_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fy = self.F(y, training=True)
            gx = self.G(x, training=True)
            g_fy = self.G(fy, training=True)
            f_gx = self.F(gx, training=True)
            
            dx_x = self.Dx(x, training=True)
            dx_fy = self.Dx(fy, training=True)
            dy_gx = self.Dy(gx, training=True)
            dy_y = self.Dy(y, training=True)

            # Discriminator losses
            dy_loss = PARAMS['gan_disc_loss'](dy_y, dy_gx)
            dx_loss = PARAMS['gan_disc_loss'](dx_x, dx_fy)

            # Generator losses
            g_loss = PARAMS['gan_gen_loss'](dy_gx)
            g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l)
            g_rec_loss = reconstruction_loss(y, gx, PARAMS['rec_loss'])

            f_loss = PARAMS['gan_gen_loss'](dx_fy)
            f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l)
            f_rec_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])

            total_f_loss = f_loss + self.b * (f_rec_loss + f_cycle_loss + self.lb*g_cycle_loss)
            total_g_loss = g_loss + self.b * (g_rec_loss + g_cycle_loss + self.lb*f_cycle_loss)

        # Calculate gradients
        f_gradients = tape.gradient(total_f_loss, self.F.trainable_variables)
        g_gradients = tape.gradient(total_g_loss, self.G.trainable_variables)

        # Apply gradients
        self.g_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        
        return {'dx_loss': dx_loss, 'dy_loss': dy_loss, 'f_loss': f_loss, 'f_rec_loss': f_rec_loss, 'f_cycle': f_cycle_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_rec_loss': g_rec_loss, 'g_cycle': g_cycle_loss}


    @tf.function
    def train_two_gen_disc_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fy = self.F(y, training=True)
            gx = self.G(x, training=True)
            g_fy = self.G(fy, training=True)
            f_gx = self.F(gx, training=True)
            
            dx_x = self.Dx(x, training=True)
            dx_fy = self.Dx(fy, training=True)
            dy_gx = self.Dy(gx, training=True)
            dy_y = self.Dy(y, training=True)

            # Discriminator losses
            dy_loss = PARAMS['gan_disc_loss'](dy_y, dy_gx)
            dx_loss = PARAMS['gan_disc_loss'](dx_x, dx_fy)

            # Generator losses
            g_loss = PARAMS['gan_gen_loss'](dy_gx)
            g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l)
            g_rec_loss = reconstruction_loss(y, gx, PARAMS['rec_loss'])

            f_loss = PARAMS['gan_gen_loss'](dx_fy)
            f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l)
            f_rec_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])

        # Calculate gradients
        dx_gradients = tape.gradient(dx_loss, self.Dx.trainable_variables)
        dy_gradients = tape.gradient(dy_loss, self.Dy.trainable_variables)

        # Apply gradients
        self.dx_optimizer.apply_gradients(zip(dx_gradients, self.Dx.trainable_variables))
        self.dy_optimizer.apply_gradients(zip(dy_gradients, self.Dy.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        
        return {'dx_loss': dx_loss, 'dy_loss': dy_loss, 'f_loss': f_loss, 'f_rec_loss': f_rec_loss, 'f_cycle': f_cycle_loss, 
                'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_rec_loss': g_rec_loss, 'g_cycle': g_cycle_loss}


    @tf.function
    def train_two_gen_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fy = self.F(y, training=True)
            gx = self.G(x, training=True)
            g_fy = self.G(fy, training=True)
            f_gx = self.F(gx, training=True)
            
            dx_x = self.Dx(x, training=True)
            dx_fy = self.Dx(fy, training=True)
            dy_gx = self.Dy(gx, training=True)
            dy_y = self.Dy(y, training=True)

            # Discriminator losses
            dy_loss = PARAMS['gan_disc_loss'](dy_y, dy_gx)
            dx_loss = PARAMS['gan_disc_loss'](dx_x, dx_fy)

            # Generator losses
            g_loss = PARAMS['gan_gen_loss'](dy_gx)
            g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l)
            g_rec_loss = reconstruction_loss(y, gx, PARAMS['rec_loss'])

            f_loss = PARAMS['gan_gen_loss'](dx_fy)
            f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l)
            f_rec_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])

            total_f_loss = f_loss + self.b * (f_rec_loss + f_cycle_loss + self.lb*g_cycle_loss)
            total_g_loss = g_loss + self.b * (g_rec_loss + g_cycle_loss + self.lb*f_cycle_loss)

        # Calculate gradients
        f_gradients = tape.gradient(total_f_loss, self.F.trainable_variables)
        g_gradients = tape.gradient(total_g_loss, self.G.trainable_variables)
        dx_gradients = tape.gradient(dx_loss, self.Dx.trainable_variables)
        dy_gradients = tape.gradient(dy_loss, self.Dy.trainable_variables)

        # Apply gradients
        self.g_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))
        self.dx_optimizer.apply_gradients(zip(dx_gradients, self.Dx.trainable_variables))
        self.dy_optimizer.apply_gradients(zip(dy_gradients, self.Dy.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        
        return {'dx_loss': dx_loss, 'dy_loss': dy_loss, 'f_loss': f_loss, 'f_rec_loss': f_rec_loss, 'f_cycle': f_cycle_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_rec_loss': g_rec_loss, 'g_cycle': g_cycle_loss}


    @tf.function
    def validate_two_gen_step(self, x, y):
        # Training=True for Normalization layer stats
        fy = self.F(y, training=True)
        gx = self.G(x, training=True)
        f_gx = self.F(gx, training=True)
        g_fy = self.G(fy, training=True)

        dx_x = self.Dx(x, training=True)
        dx_fy = self.Dx(fy, training=True)
        dy_gx = self.Dy(gx, training=True)
        dy_y = self.Dy(y, training=True)

        # dx_loss = dist_ls_gan_loss_disc(dx_x, dx_fy)
        f_loss = reconstruction_loss(x, fy, dist_mae_loss)
        f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l) 

        # dy_loss = dist_ls_gan_loss_disc(dy_y, dy_gx)
        g_loss = reconstruction_loss(y, gx, dist_mae_loss)
        g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l) 

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'val_f_rec_loss': f_loss, 'val_f_cycle': f_cycle_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_rec_loss': g_loss, 'val_g_cycle': g_cycle_loss}


    @tf.function
    def train_two_gen_no_disc_step(self, x, y):
        with tf.GradientTape() as g_tape, tf.GradientTape() as f_tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fy = self.F(y, training=True)
            gx = self.G(x, training=True)
            g_fy = self.G(fy, training=True)
            f_gx = self.F(gx, training=True)

            g_loss = reconstruction_loss(y, gx, PARAMS['rec_loss'])
            g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l) 
            
            f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
            f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l) 
            
            total_g_loss = g_loss + g_cycle_loss + self.lb*f_cycle_loss
            total_f_loss = f_loss + f_cycle_loss + self.lb*g_cycle_loss

        # Calculate gradients
        f_gradients = f_tape.gradient(total_f_loss, self.F.trainable_variables)
        g_gradients = g_tape.gradient(total_g_loss, self.G.trainable_variables)

        # Apply gradients
        self.g_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))

        # Calculate metrics
        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'f_loss': f_loss, 'f_cycle': f_cycle_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_cycle': g_cycle_loss}


    @tf.function
    def validate_two_gen_no_disc_step(self, x, y):
        # Training=True for Normalization layer stats
        fy = self.F(y, training=True)
        gx = self.G(x, training=True)
        f_gx = self.F(gx, training=True)
        g_fy = self.G(fy, training=True)

        f_loss = reconstruction_loss(x, fy, dist_mae_loss)
        f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l) 

        g_loss = reconstruction_loss(y, gx, dist_mae_loss)
        g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l) 

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'val_f_loss': f_loss, 'val_f_cycle': f_cycle_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_loss': g_loss, 'val_g_cycle': g_cycle_loss}


    @tf.function
    def train_one_gen_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            # Generator F translates Y -> X.
            fy = self.F(y, training=True)

            dx_x = self.Dx(x, training=True)
            dx_fy = self.Dx(fy, training=True)
            dx_loss = PARAMS['gan_disc_loss'](dx_x, dx_fy)

            f_loss = PARAMS['gan_gen_loss'](dx_fy)
            f_rec_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
            total_f_loss = f_rec_loss + 1e-3*f_loss
        # Calculate gradients
        f_gradients = tape.gradient(total_f_loss, self.F.trainable_variables)
        dx_gradients = tape.gradient(dx_loss, self.Dx.trainable_variables)
        # Apply gradients
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))
        self.dx_optimizer.apply_gradients(zip(dx_gradients, self.Dx.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        return {'f_loss': f_loss, 'f_rec_loss': f_rec_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr, 'dx_loss': dx_loss}

    @tf.function
    def validate_one_gen_step(self, x, y):
        fy = self.F(y, training=False)
        f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        return {'val_f_rec_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr}


    @tf.function
    def train_one_gen_no_disc_step(self, x, y):
        with tf.GradientTape() as f_tape:
            # Generator F translates Y -> X.
            fy = self.F(y, training=True)
            f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
        # Calculate gradients
        f_gradients = f_tape.gradient(f_loss, self.F.trainable_variables)
        # Apply gradients
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'f_loss': f_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr}

    @tf.function
    def validate_one_gen_no_disc_step(self, x, y):
        # Training=True for Normalization layer stats
        fy = self.F(y, training=True)
        f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        return {'val_f_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr}


    @tf.function
    def train_unet_step(self, x, y):
        with tf.GradientTape() as f_tape:
            # Generator F translates Y -> X.
            fy = self.F(y, training=True)
            f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
        # Calculate gradients
        f_gradients = f_tape.gradient(f_loss, self.F.trainable_variables)
        # Apply gradients
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'f_loss': f_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr}

    @tf.function
    def validate_unet_step(self, x, y):
        fy = self.F(y, training=False)
        f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'val_f_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr}


def main():
    train = Train()
    train.train()


if __name__ == '__main__':
    main()
