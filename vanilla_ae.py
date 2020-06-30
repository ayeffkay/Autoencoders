import numpy as np
import torch.nn as nn
from base_ae import Autoencoder
from modules import RestoreSize, ModelOutput


class VanillaAutoencoder(Autoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        flat_inp = np.prod(self.input_size)
        hid = kwargs['hid']

        self.encoder = nn.Sequential(nn.Flatten(),
                            nn.Linear(flat_inp, hid[0]), 
                            nn.BatchNorm1d(hid[0]), 
                            nn.LeakyReLU(), 
                            nn.Linear(hid[0], hid[1]), 
                            nn.BatchNorm1d(hid[1]), 
                            nn.LeakyReLU())
        
        self.latent_repr = nn.Linear(hid[1], self.latent_dim)

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, hid[1]), 
                            nn.BatchNorm1d(hid[1]), 
                            nn.LeakyReLU(), 
                            nn.Linear(hid[1], hid[0]), 
                            nn.BatchNorm1d(hid[0]), 
                            nn.LeakyReLU(), 
                            nn.Linear(hid[0], flat_inp), 
                            nn.Sigmoid(), 
                            RestoreSize(self.input_size))
        
        
    def encode(self, x, **kwargs):
        enc_out = self.encoder(x)
        latent_code = self.latent_repr(enc_out)
        return latent_code,
    

    def decode(self, x, **kwargs):
        out = self.decoder(x)
        return out
        
        
    def forward(self, x, **kwargs):
        latent_code = self.encode(x)[0]
        reconstruction = self.decode(latent_code)

        return ModelOutput(latent_code, reconstruction)
