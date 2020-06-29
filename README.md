Encoding people's faces with autoencoders. All experiments were conducted on [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)

**Files description**

* gifs/* - animated reconstruction histories for validation images
* config.py -- various parameters (eg., image size, model architectures)
* get_dataset.py -- helper module for downloading and preprocessing data
* data_utils.py -- further preprocessing, sampling and plotting functions
* modules.py -- helper modules for restoring size, padding and unified output for autoencoders
* losses.py -- redefined MSE for unified training and VAE loss
* train_util.py -- model training while saving losses and reconstructions history; animation generation, t-SNE projections and search for similar images are also implemented here 
* base_ae.py -- abstract base class from which other autoencoders inherit to
* vanilla_ae.py -- simple autoencoder with few dense layers
* vae.py - convolutional VAE and CVAE (two in one)
* demo.ipynb -- AE, VAE and CVAE demo with applications
