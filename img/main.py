# pip install torch torchvision clip-by-openai
import clip
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CLIP 官方输入预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711),
                             ),
    ])

    # 加载 MNIST 数据集
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=preprocess)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=preprocess)

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    # 加载 CLIP 模型
    model, _ = clip.load('ViT-L/14', device=device)

    # 特征提取函数
    def extract_features(dataloader, model):

        model.eval()
        features = []
        labels = []

        with torch.no_grad():

            for images, label in tqdm(dataloader, ncols=100):
                images = images.to(device)
                image_features = model.encode_image(images)
                features.append(image_features)
                labels.append(label)

        return torch.cat(features), torch.cat(labels)


    # 提取 MNIST 特征
    train_features, train_labels = extract_features(train_loader, model)
    test_features, test_labels = extract_features(test_loader, model)

    print('train_features.shape =', train_features.shape)
    print('train_labels.shape =', train_labels.shape)
    print()
    print('test_features.shape =', test_features.shape)
    print('test_labels.shape =', test_labels.shape)
