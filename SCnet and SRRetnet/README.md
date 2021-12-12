# Setup
1. Upload data to 'datasets' folder. Assumes images are jpegs. I have the following file structure
    - **4 layer cultured neurons**: `./datasets/cannula-4-layer/layer1/ccm-jpeg/0.jpeg` and `./datasets/cannula-4-layer/layer1/ref-jpeg/0.jpeg`. \
    Each layer (layer1,layer2,layer3,layer4) has directories 'ccm-jpeg' and 'ref-jpeg'. 
    - **Brain**: `datasets/ccm-jpeg/0.jpeg`. \
    If you use a different file structure you will need to edit Data.py `setup_datasets()`.
2. Create directory 'figures'

## Required Packages 
If no version is specified any should work
- Tensorflow (version 2.3.0)
- Tensorflow Addons (version 0.11.2)
- Numpy
- Matplotlib
- Scikit-image

# Train on 4 layer cultured neurons and test on brain samples
1. Configure Train.py PARAMS.
2. Run `python Train.py`

Best results on the 4 layer cultured neurons are with following PARAMS configuration:
```
PARAMS = {
    'dir': '4_layer', 'num_channels': 1, 'raw_input_shape': (128,128), 'target_input_shape': (128,128),

    'generator': cycle_reconstructor, 
    'rec_loss': dist_mae_loss,
    'cycle_loss': dist_mae_loss,
    'gan_gen_loss': dist_ls_gan_loss_gen,
    'gan_disc_loss': dist_ls_gan_loss_disc,


    'chpt': get_and_increment_chpt_num(),
    'channel_format': 'NCHW',
    'load_chpt': None,
    'save_models': True,

    'epochs': 45,
    'batch_size': 8,
    'learning_rate': 2e-4,
    'disc_lr': 2e-5,

    # Loss scalars:
        # type double: Loss = GAN + beta(MAE + lambda(Forward + lambda_b*Backward))
    'lambda': 1,
    'lambda_b': 1,
    'beta': 100,

    'generator_to_discriminator_steps': 0, 

    'type': 'double', # Self Consistent Supervised

    'F_PARAMS': {
        'filters': {'down': [64, 128, 256, 256], 'up': [256, 128, 64]},
        'dropout': {'down': [0.0, 0.0, 0.0, 0.0], 'up': [0.0, 0.0, 0.0]},
        'kernels': [3,3], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'instance',
        'activation': 'relu',
        'skip': False,
    },
    'G_PARAMS': {
        'filters': {'down':[64, 128, 256, 256], 'up':[256, 128, 64]},
        'dropout': {'down':[0.0, 0.0, 0.0, 0.0], 'up':[0.0, 0.0, 0.0]},
        'kernels': [3,3], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'instance',
        'activation': 'relu',
        'skip': False,
    },
}
```


# CCM Image Generation and Evaluation from presaved G Network
1. Configure GenerateCCM.py PARAMS.
2. Run `python GenerateCCM.py`
