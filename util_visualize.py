import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d
#%matplotlib notebook


def show_2D_XXX(cls, latent_vecs, target, title, figsize, labels):
    latent_vecs = latent_vecs
    latent_vecs_reduced = cls(n_components=2, random_state=0).fit_transform(latent_vecs)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cax = ax.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1], c=target, cmap='jet')
    ax.set_title(title)
    cbar = fig.colorbar(cax)
    if labels is not None:
        cbar.ax.set_yticklabels(labels)
    plt.show()


def show_2D_tSNE(latent_vecs, target, title='t-SNE viz', figsize=(8, 6), labels=None):
    show_2D_XXX(TSNE, latent_vecs, target, title, figsize, labels)


def show_2D_PCA(latent_vecs, target, title='PCA viz', figsize=(8, 6), labels=None):
    show_2D_XXX(PCA, latent_vecs, target, title, figsize, labels)


def show_3D_tSNE(latent_vecs, target, title='3D t-SNE viz', figsize=(13,10), labels=None):
    latent_vecs = latent_vecs
    tsne = TSNE(n_components=3, random_state=0).fit_transform(latent_vecs)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=target, cmap='jet')
    ax.set_title(title)
    cbar = fig.colorbar(scatter)
    if labels is not None:
        cbar.ax.set_yticklabels(labels)
    plt.show()
