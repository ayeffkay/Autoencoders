import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_size = kwargs['input_size']
        self.latent_dim = kwargs['latent_dim']
        self.device = kwargs['device']

        self.register_buffer('mu0', torch.zeros((kwargs['latent_dim'])))
        self.register_buffer('sigma0', torch.ones((kwargs['latent_dim'])))

        self.normal_distribution = torch.distributions.normal.Normal(
            loc=self.mu0,
            scale=self.sigma0
        )

        self.encoder = None
        self.decoder = None

    def encode(self, x, **kwargs):
        pass

    def decode(self, x, **kwargs):
        pass

    def forward(self, x, **kwargs):
        pass

    def sample(self, n, **kwargs):
        rand = self.normal_distribution.sample([n]).to(self.device)
        out = self.decode(x=rand, label=kwargs.get('label', None))
        return out

    def add_feature(self, x1, x2):
        """"
          add feature from x2 to x1 (e.g., add smile, hat, bangs, etc.)
        """
        self.eval()
        with torch.no_grad():
            latent_code1 = self.encode(x1)[0]
            latent_code2 = self.encode(x2)[0]

            feat1 = torch.mean(latent_code1, dim=0)
            feat2 = torch.mean(latent_code2, dim=0)
            diff = feat2 - feat1

            out = self.decode(latent_code1 + diff)
        return out

    def morphing(self, x1, x2, alpha=0.5):
        self.eval()
        with torch.no_grad():
            latent_code1 = self.encode(x1)[0]
            latent_code2 = self.encode(x2)[0]
            latent_code = alpha * latent_code1 + (1 - alpha) * latent_code2

            out = self.decode(latent_code)
        return out
