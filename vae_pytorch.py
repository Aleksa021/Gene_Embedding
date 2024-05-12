from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from copy import deepcopy
from datetime import datetime
import os
import pathlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
from anndata.experimental.pytorch import AnnLoader
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
torch.manual_seed(0)



class VAE(nn.Module):
    def __init__(self,input_size, hidden_size,laten_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, laten_size)
        self.fc22 = nn.Linear(hidden_size, laten_size)
        self.fc3 = nn.Linear(laten_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar




# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, loader, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(loader):
        data = batch.X
        data = data.to('cuda')
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % LOG_INTERVAL == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(loader.dataset),
        #         100. * batch_idx / len(loader),
        #         loss.item() / len(data)))

    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #       epoch, train_loss / len(loader.dataset)))


def validation(epoch,loader,model):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        data = batch.X
        y = batch.obs['cell type'].data.cpu().numpy()
        data = data.to('cuda')
        mu= model(data)[1]
        data_2d = PCA(n_components=2).fit_transform(mu.data.cpu().numpy())
        test_score = silhouette_score(data_2d,y)
        # print("2d ", silhouette_score(data_2d,y), 'latent ', silhouette_score(mu.data.cpu().numpy(),y))
        # df = pd.DataFrame(np.concatenate([data_2d,y[:,None]],axis=-1), columns =['PCA 1','PCA 2','cell type'])
        # df['cell type'] = df['cell type'].astype('category')
    # test_loss /= len(test_loader.dataset)
    print('====> Validation set score: {:.4f}'.format(test_score))
    return test_score

if __name__ == "__main__":

    data = sc.read_h5ad('data/stereo_seq_olfactory_bulb_bin140_annotation.h5ad')
    n_cells,n_genes = data.X.shape
    encoder_celltype = LabelEncoder()
    encoder_celltype.fit(data.obs['cell type'])
    train_ind, test_ind = next(iter(StratifiedShuffleSplit(n_splits=2,train_size=0.8,random_state=11).split(data, data.obs['cell type'])))
    encoders = {
        'obs': {
            'cell type': encoder_celltype.transform
        },
        "X": MinMaxScaler(clip=True).fit(data[train_ind].X).transform
    }
  
    HIDDEN_DIM = 1024
    K_FOLDS = 5
    LATENT_DIMS = [10,100,500,1000,2000]
    N_EPOCHS_CV = 100
    N_EPOCHS = 500
    BATCH_SIZE = 128
    USE_CUDA  = True
    LOG_INTERVAL = 1
    test_loader = AnnLoader(data[test_ind], batch_size=len(test_ind), shuffle=True, convert=encoders, use_cuda=USE_CUDA)
    train_data = data[train_ind]
    val_scores = {dim: [[] for _ in range(K_FOLDS)] for dim in LATENT_DIMS}

    # FINDING OPTIMAL LATENT DIMENSION
    for latent_dim in LATENT_DIMS:
        cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=0)
        for fold_index, (train_ind,val_ind) in enumerate(cv.split(train_data,train_data.obs['cell type'])):
            print(f'Stating fold {fold_index}, latent_dim={latent_dim}')
            model = VAE(n_genes, HIDDEN_DIM, latent_dim).to('cuda')
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            train_loader = AnnLoader(train_data[train_ind], batch_size=BATCH_SIZE, shuffle=True, convert=encoders, use_cuda=USE_CUDA)
            val_loader = AnnLoader(train_data[val_ind], batch_size=len(train_data[val_ind]), shuffle=True, convert=encoders, use_cuda=USE_CUDA)
            for epoch in range(1, N_EPOCHS_CV + 1):
                train(epoch,train_loader,model,optimizer)
                val_scores[latent_dim][fold_index].append(validation(epoch, val_loader, model))
        
        
    print(val_scores)
    cv_score = {key: 0 for key in val_scores}
    for latent_dim, scores in val_scores.items():
        cv_score[latent_dim] = np.array(scores).max(axis=1).mean()
    print(cv_score)
    best_latent_dim = max(cv_score, key=cv_score.get)
    ## REFITING MODEL WITH OPTIMAL LATENT DIMENSION
    train_ind, test_ind = next(iter(StratifiedShuffleSplit(n_splits=2,train_size=0.8,random_state=0).split(train_data, train_data.obs['cell type'])))
    model = VAE(n_genes, HIDDEN_DIM, best_latent_dim).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = AnnLoader(train_data[train_ind], batch_size=BATCH_SIZE, shuffle=True, convert=encoders, use_cuda=USE_CUDA)
    val_loader = AnnLoader(train_data[val_ind], batch_size=len(train_data[val_ind]), shuffle=True, convert=encoders, use_cuda=USE_CUDA)
    best_val_score = -1
    for epoch in range(1, N_EPOCHS + 1):
        train(epoch,train_loader,model,optimizer)
        val_score = validation(epoch, val_loader, model)
        if val_score > best_val_score:
            best_model = deepcopy(model)
            best_val_score = val_score

    ## SAVING MODEL
    current_datetime = datetime.now()
    date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    path = pathlib.Path('models',date_time_string)
    os.makedirs(path, exist_ok=True)
    torch.save(best_model, path / 'model.pth')
    ## EVALUATION 
    test_data = next(iter(test_loader))
    X = test_data.X
    y = test_data.obs['cell type'].data.cpu().numpy()
    means = best_model.encode(test_data.X)[0].data.cpu().numpy() # get mean values of the latent variables
    print(silhouette_score(means, y))
    test_data = test_data.to_adata()
    test_data.obsm['X_vae'] = means

    sc.pp.pca(test_data, n_comps=2)
    sc.pl.pca(test_data, color='cell type', wspace=0.35, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 4.8)
    plt.tight_layout()
    plt.savefig(path/'pca.png')

    sc.pp.neighbors(test_data, use_rep='X_vae')
    sc.tl.umap(test_data)
    sc.pl.umap(test_data, color='cell type', wspace=0.35, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 4.8)
    plt.tight_layout()
    plt.savefig(path/'umap.png')