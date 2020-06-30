import torch
from copy import deepcopy
import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import config
from modules import ModelOutput
import data_utils

        
class TrainUtility:
    def __init__(self, model, dataloaders, criterion, optimizer, img_size=config.input_size, epochs=1, train=True):
        assert set(dataloaders.keys()).issubset({'train', 'valid'})

        # [C, H, W]
        self.img_size = img_size

        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch_loss = 0
        self.train_hist = {mode: [] for mode in self.dataloaders.keys()}

        # images to save for animated history
        self.n_anim = config.n_anim
        self.img_hist = []
        self.latent_codes = dict.fromkeys(dataloaders.keys())

        if train:
            self.train(epochs)


    def run_epoch(self, data, mode='valid', return_out=False):
        self.epoch_loss = 0
        epoch_out = dict.fromkeys(['feature', 'code'])

        with torch.set_grad_enabled(mode=='train'):
            self.model.train() if mode == 'train' else self.model.eval()

            for i, batch in enumerate(data):
                out = self.model(x=batch['feature'], label=batch['label'])
                if return_out:
                    try:
                        epoch_out['feature'] = torch.cat((epoch_out['feature'], batch['feature']), dim=0)
                        epoch_out['code'] = torch.cat((epoch_out['code'], out.latent_code), dim=0)
                    except:
                        epoch_out['feature'] = batch['feature']
                        epoch_out['code'] = out.latent_code

                loss = self.criterion(out.rec, batch['loss_img'], out.mu, out.log_sigma)
                self.epoch_loss += loss.item()

                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if mode == 'valid' and i == 0:
                    cur = torch.cat((batch['feature'][:self.n_anim], out.rec[:self.n_anim]), dim=0)
                    union = self.concat_images(cur, show=True)
                    self.img_hist += [union]

            self.epoch_loss /= len(data)
        return epoch_out


    def train(self, epochs=1):
        min_loss = 1e+5
        best_model_wts = deepcopy(self.model.state_dict())

        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                for mode, data in self.dataloaders.items():
                    self.run_epoch(data, mode, epoch)
                    self.train_hist[mode] += [self.epoch_loss]
                    if mode == 'valid':
                        pbar.write('|| epoch loss: {:.4f}'.format(self.epoch_loss))
                        if self.epoch_loss < min_loss:
                            min_loss = self.epoch_loss
                            best_model_wts = deepcopy(self.model.state_dict())
                pbar.update(1)
        self.model.load_state_dict(best_model_wts)


    def concat_images(self, images, n_row=2, n_col=10, show=False, cmap='gray', title=None):
        n_channels, h, w = self.img_size
        out = np.ones((h * n_row, w * n_col, n_channels))
        for i in range(n_row * n_col):
            if i >= len(images):
                break

            if isinstance(images[i], torch.Tensor):
                img = data_utils.tensor_to_np(images[i])
            else:
                img = images[i]
            
            row = i // n_col
            col = i % n_col

            out[row*h:(row+1)*h, col*w:(col+1)*w, :] = img

        if show:
            data_utils.plot_gallery([out], n_row=1, n_col=1, figsize=(10, 8))
            
        return out


    def make_anim(self):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
        frames = len(self.img_hist)
        im = ax.imshow(np.ones_like(self.img_hist[0]), animated=True)

        ax.set_title('Epoch 0')
        ax.grid(False)
        ax.axis('off')
        frame = 0

        def update(frame):
            im.set_array(self.img_hist[frame])
            im.axes.set_title(f'Epoch:{frame + 1}')
            return im,

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
        anim.save(f'{self.model.__class__.__name__}_history.gif', 
                  dpi=80, writer='imagemagick')
        return anim


    def plot_history(self):
        fig, ax = plt.subplots()
        fig.suptitle('Training history')
        for mode, loss in self.train_hist.items():
            epochs = np.arange(len(self.train_hist[mode])) + 1
            ax.plot(epochs, self.train_hist[mode], label=f'{mode}_loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()


    def extract_latent_codes(self, dl_name='valid'):
        out = self.run_epoch(self.dataloaders[dl_name], mode='test', return_out=True)
        feature, code = [x for key, x in out.items()]
        return feature, code


    def get_labels(self, dl_name='valid'):
        labels = [batch['label'].cpu().numpy() for batch in self.dataloaders[dl_name]]
        return np.vstack(labels)


    def get_2d_embeddings(self, dl_name='valid', show=True):
        self.latent_codes[dl_name] = self.extract_latent_codes(dl_name)[1]
        labels = self.get_labels(dl_name)

        tsne = TSNE(n_components=2)
        embeddings = tsne.fit_transform(self.latent_codes[dl_name].cpu().numpy())
        if show:
            self.plot_2d_embeddings(embeddings, labels)
        return embeddings, labels


    @staticmethod
    def plot_2d_embeddings(embeddings, labels):
        plt.title('t-SNE projection')
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='Set1', alpha=0.5)
    
    
    def get_similar(self, img, n_neighbors=5, show=True):
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        dl_name = 'train'
        x, self.latent_codes[dl_name] = self.extract_latent_codes(dl_name)
        nn.fit(self.latent_codes[dl_name].cpu().numpy())

        if isinstance(img, np.ndarray):
            img = data_utils.np_to_tensor(img)
        if len(img.size()) == 3:
            img = img.unsqueeze(0)
        img = img.to(self.model.device)

        with torch.no_grad():
            img_code = self.model.encode(img)[0].cpu().numpy()

        dist, ind = nn.kneighbors(img_code, n_neighbors=n_neighbors, return_distance=True)
        dist = dist.squeeze(0)
        ind = ind.squeeze(0)
        if show:
            dist = list(map(lambda d: 'Dist.={:.3f}'.format(d), dist))
            data_utils.plot_gallery(torch.cat([img, x[ind]], dim=0), n_row=1, n_col=n_neighbors + 1, 
                         subplot_titles = ['Original image'] + dist)
        return dist, x[ind]
