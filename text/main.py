import txt_reader
import clip_feature

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    bible = txt_reader.read_txt('bible.txt')
    literature = txt_reader.read_txt('don-quixote.txt')
    literature += txt_reader.read_txt('iliad.txt')
    literature += txt_reader.read_txt('odyssey.txt')

    print('len(bible) =', len(bible))
    print('len(literature) =', len(literature))

    bible_feature = clip_feature.extract_text_features(bible)
    literature_feature = clip_feature.extract_text_features(literature)

    label = [0] * len(bible_feature) + [1] * len(literature_feature)
    data = np.concatenate((bible_feature, literature_feature), axis=0)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # 可视化数据
    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=label,
                marker='.', cmap='bwr', alpha=0.5)

    plt.title('PCA Visualization of Text Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    # plt.show()

    # 使用t-SNE进行降维到二维
    tsne = TSNE(n_components=2, n_jobs=-1)
    data_tsne = tsne.fit_transform(data)

    # 可视化数据
    plt.figure(figsize=(8, 6))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=label,
                marker='.', cmap='bwr', alpha=0.5)

    plt.title('t-SNE Visualization of Text Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid()

    np.savez_compressed('data.npz',
                        data=data, label=label,
                        data_pca=data_pca, data_tsne=data_tsne,
                        )
    plt.show()
