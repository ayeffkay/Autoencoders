import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules import RestoreSize, ModelOutput
from base_ae import Autoencoder


class VariationalAutoencoder(Autoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        """VAE and CVAE (kwargs['n_classes'] required)"""
        
        restore_size = kwargs['restore_size']
        self.restore = RestoreSize(restore_size)
        # for CVAE
        self.n_classes = kwargs.get('n_classes', 0)

        if self.n_classes:
            if self.n_classes == 2:
                self.n_classes -= 1 # one unit is enough to encode two classes
            hw = np.prod(self.input_size[1:])
            self.condition_enc = nn.Sequential(nn.Linear(self.n_classes, hw // 4), 
                                                nn.LeakyReLU(),
                                                nn.Linear(hw // 4, hw), 
                                                nn.LeakyReLU())

        self.encoder = self.build_enc_dec(kwargs['enc_cfg'])
        self.dec_inp = nn.Linear(self.latent_dim + self.n_classes, np.prod(restore_size))
        self.decoder = self.build_enc_dec(kwargs['dec_cfg'], enc=False)

        self.mu_repr = nn.Linear(np.prod(restore_size), self.latent_dim)
        self.log_sigma_repr = nn.Linear(np.prod(restore_size), self.latent_dim)
                
                
    def build_enc_dec(self, config, enc=True):
        layers = nn.ModuleList()
        for layer_cfg in config:
            if enc:
                layers.append(self.enc_block(*layer_cfg))
            else:
                layers.append(self.dec_block(*layer_cfg))
        return layers


    @staticmethod
    def enc_block(channels, kernel, stride, padding, n=1):
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(channels[i], channels[i + 1], kernel[i], stride[i], padding[i])]
            layers += [nn.BatchNorm2d(channels[i + 1])]
        layers += [nn.LeakyReLU()]
        return nn.Sequential(*layers)


    @staticmethod
    def dec_block(channels, kernel, stride, padding, n=1, last=False):
        layers = [nn.ConvTranspose2d(channels[0], channels[1], kernel[0], stride[0], padding[0]),
                nn.BatchNorm2d(channels[1])]
        for i in range(1, n):
            layers += [nn.Conv2d(channels[i], channels[i + 1], kernel[i], stride[i], padding[i])]
            if i == n - 1 and last:
                layers += [nn.Sigmoid()]
            else:
                layers += [nn.BatchNorm2d(channels[i + 1])]
        if not last:
            layers += [nn.LeakyReLU()]
        return nn.Sequential(*layers)

        
    def encode(self, x, **kwargs):
        if self.n_classes >= 1:
            b, c, h, w = x.size()
            condition = self.condition_enc(kwargs['label']).view(b, h, w).unsqueeze(1)
            x = torch.cat((x, condition), dim=1) # concat by channels
        for layer in self.encoder:
            x = layer(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        mu = self.mu_repr(x)
        log_sigma = self.log_sigma_repr(x)
        latent_code = self.gaussian_sampler(mu, log_sigma)
        
        return latent_code, mu, log_sigma,


    # reparametrization
    def gaussian_sampler(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        # sample latent_dim batch times
        eps = self.normal_distribution.sample([std.size(0)]).to(self.device)
        return eps * std + mu


    def decode(self, x, **kwargs):
        if self.n_classes >= 1:
            label = kwargs['label']
            x = torch.cat((x, label), dim=1)
        x = self.restore(self.dec_inp(x))
        for layer in self.decoder:
            x = layer(x)
        return x
        
        
    def forward(self, x, **kwargs):
        latent_code, mu, log_sigma = self.encode(x.clone(), **kwargs)
        reconstruction = self.decode(latent_code, **kwargs)

        return ModelOutput(latent_code, reconstruction, mu, log_sigma)
