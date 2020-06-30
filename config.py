import random
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


img_h = 64
img_w = 64
n_channels = 3
image_size = (img_h, img_w, n_channels)
input_size = (n_channels, img_h, img_w)

train_size = 10000
batch_size = 32
# images to save for animated history
n_anim = 10

# vanilla autoencoder config
ae_latent_dim = 100
ae_hid = [1024, 512]

# VAE and CVAE config
vae_latent_dim = 256
# channels, kernels, strides, paddings, n_blocks, is_last (for decoder only)
enc_cfg_vae = [[[3, 16, 16], [3, 2], [1, 2], [1, 0], 2], 
               [[16, 32, 32], [3, 2], [1, 2], [1, 0], 2],
               [[32, 64, 64], [3, 2], [1, 2], [1, 0], 2],
               [[64, 128, 128], [3, 2], [1, 2], [1, 0], 2],
               ]
enc_cfg_cvae = [[[4, 16, 16], [3, 2], [1, 2], [1, 0], 2], 
               [[16, 32, 32], [3, 2], [1, 2], [1, 0], 2],
               [[32, 64, 64], [3, 2], [1, 2], [1, 0], 2],
               [[64, 128, 128], [3, 2], [1, 2], [1, 0], 2],
               ]
dec_cfg_vae = [[[128, 128, 64], [2, 3], [2, 1], [0, 1], 2, 0],
               [[64, 64, 32], [2, 3], [2, 1], [0, 1], 2, 0],
               [[32, 32, 16], [2, 3], [2, 1], [0, 1], 2, 0],
               [[16, 16, 3], [2, 3], [2, 1], [0, 1], 2, 1]]
restore_size_vae = [128, 4, 4]

