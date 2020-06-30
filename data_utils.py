import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from IPython.display import clear_output
from skimage.draw import rectangle
from sklearn import preprocessing
import config


def np_to_tensor(img, add_batch_dim=True):
    assert len(img.shape) in [3, 4] and isinstance(img, np.ndarray)
    if len(img.shape) == 4:
        new_axis = [0, 3, 1, 2] 
        img = img.transpose(new_axis)
    elif len(img.shape) == 3:
        new_axis = [2, 0, 1]
        img = img.transpose(new_axis)
        if add_batch_dim:
            img = np.expand_dims(img, axis=0)
    return torch.FloatTensor(img)


def tensor_to_np(img, batch_first=True):
    assert len(img.size()) in [3, 4] and isinstance(img, torch.Tensor)
    if len(img.size()) == 4:
        img = img.data.permute(0, 2, 3, 1)
    elif len(img.size()) == 3:
        img = img.data.permute(1, 2, 0)
    return img.squeeze().cpu().numpy()


def plot_gallery(images, n_row=2, n_col=10, cmap='gray', figsize=None, 
                 title=None, subplot_titles=None):
    clear_output(wait=True)
    
    if figsize is None:
        figsize = (1.3 * n_col, 1.5 * n_row)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    fig.suptitle(title)

    for i in range(n_row * n_col):
        if i >= len(images):
            break

        ax = fig.add_subplot(n_row, n_col, i + 1)
        if subplot_titles is not None:
            ax.title.set_text(subplot_titles[i])

        if isinstance(images[i], torch.Tensor):
            img = tensor_to_np(images[i])
        else:
            img = images[i]
        
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
    plt.show()
    
    
def train_test_split_seq(data, train_size=config.train_size):
    train_idx = np.arange(train_size)
    val_size = data.shape[0] - train_size
    val_idx = np.arange(train_size, train_size + val_size)
    X_train = data[train_idx]
    X_val = data[val_idx]
    return X_train, X_val, train_idx, val_idx


def get_labels_from_attrs(attrs, colnames, threshold=0):
    subset = attrs[colnames].values
    if len(colnames) > 1:
        labels = np.argmax(subset, axis=1)
    else:
        labels = (subset > threshold).astype(int)

    binarizer = preprocessing.LabelBinarizer()
    binarizer.fit(labels)
    return labels, binarizer



class FaceDataset(Dataset):
    def __init__(self, data, img_size=config.image_size, device=config.device, **kwargs):
        """
            Generate dataset with specified attributes
            kwargs['attr'] -- attribute description in the form 
            {'label': ndarray, 'binarizer': LabelBinarizer}
            kwargs['noise'] -- add noise with factor
            kwargs['occlusion'] -- generate occlusion of shape [H, W]
        """
        super().__init__()
        self.data = data
        self.device = device
        self.h, self.w = img_size[:2]

        if len(kwargs) > 0:

            if kwargs.get('attr', None) and isinstance(kwargs['attr'], dict):
                self.binarizer = kwargs['attr'].get('binarizer', None)
                self.labels = kwargs['attr'].get('labels', None)
                assert self.binarizer is not None and self.labels is not None
                self.idx = {cl: np.where(self.labels == cl)[0] for cl in self.binarizer.classes_}
              
            if kwargs.get('noise', None):
                self.noise_factor = kwargs['noise']

            if kwargs.get('occlusion', None):
                self.occlusion = kwargs['occlusion']
                self.x_o = np.random.randint(0, self.h - self.occlusion[0], len(self.data))
                self.y_o = np.random.randint(0, self.w - self.occlusion[1], len(self.data))


    def __len__(self):
        return len(self.data)

    def draw_occlusion(self, img, i):
        x = self.x_o[i]
        y = self.y_o[i]
        rr, cc = rectangle(start=(x, y), extent=self.occlusion)
        img[:, rr, cc] = 1
        return img

    def __getitem__(self, i):
        item = dict()
        img = torch.FloatTensor(self.data[i]).permute(2, 0, 1)

        if hasattr(self, 'noise_factor'):
            noisy = torch.clamp(img + self.noise_factor * torch.randn_like(img), 0, 1)
            item['feature'] = noisy.to(self.device)

        elif hasattr(self, 'occlusion'):
            occluded = self.draw_occlusion(img.clone(), i)
            item['feature'] = occluded.to(self.device)
        
        else:
            item['feature'] = img.to(self.device)

        item['loss_img'] = img.to(self.device)

        if hasattr(self, 'binarizer') and hasattr(self, 'labels'):
            label = self.binarizer.transform([self.labels[i]])[0]
            item['label'] = torch.FloatTensor(label).to(self.device)
        else:
            item['label'] = 0

        return item

    def sample_by_condition(self, n=32):
        assert hasattr(self, 'idx')

        samples = dict.fromkeys(self.idx.keys())

        for cl, idx in self.idx.items():
            np.random.shuffle(idx)
            one_cl = [self.__getitem__(idx[i])['feature'].unsqueeze(0) for i in range(n)]
            samples[cl] = torch.cat(one_cl, dim=0)
        return samples
