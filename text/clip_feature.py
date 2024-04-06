import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def extract_text_features(texts, batch_size=64):
    """
    分批从给定文本中提取特征

    参数:
        texts (list of str): 要从中提取特征的文本列表
        model (CLIPModel): 预加载的CLIP模型
        processor (CLIPProcessor): 预加载的CLIP处理器
        batch_size (int): 每个批次的文本数量

    返回:
        features (torch.Tensor): 所有文本的提取特征
    """
    model_name = 'openai/clip-vit-large-patch14'
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 确保模型在正确的设备上
    device = 'cuda'
    model.to(device)

    all_features = []
    for i in tqdm(range(0, len(texts), batch_size), desc='clip', ncols=100):

        batch_texts = texts[i:i + batch_size]
        inputs = processor(text=batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            all_features.append(outputs.cpu())  # 将特征移回CPU

    return torch.cat(all_features, dim=0)


if __name__ == '__main__':

    pass
