import string
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def load_word_vectors(filename):
    word_vectors = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors


# 加载词向量
word_vectors = load_word_vectors('word2vec/test_captions.json_word_vectors.txt')  # 替换为您的词向量文件路径


def text_to_vectors(text, word_vectors):
    # 转换为小写并移除标点
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    # 获取词向量的维度
    vector_dim = len(next(iter(word_vectors.values())))

    # 创建开始符和结束符的词向量
    start_vector = np.ones(vector_dim)
    end_vector = np.full(vector_dim, 2)

    # 将开始符、词向量、结束符合并
    vectors = [start_vector] + [word_vectors.get(word, np.zeros(vector_dim)) for word in words] + [end_vector]
    return np.array(vectors)


class ImageTextDataset(Dataset):
    def __init__(self, json_file_path, image_folder_path, word_vectors):
        """
            Args:
               json_file_path (string): JSON文件的路径，包含图片名称和对应的描述。
               image_folder_path (string): 包含图片的文件夹路径。
        """
        with open(json_file_path, 'r', encoding='utf-8') as file:
            self.descriptions = json.load(file)
        self.image_folder_path = image_folder_path
        self.word_vectors = word_vectors

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        image_name, description = list(self.descriptions.items())[idx]
        image_path = os.path.join(self.image_folder_path, image_name)

        # 加载图像并调整尺寸
        image = Image.open(image_path)
        image = image.resize((512, 512))  # 将图像大小调整为 750x1101

        # 转换图像为numpy数组，然后转为Tensor
        image = np.array(image)
        if image.shape[2] == 4:  # 检查是否有alpha通道
            image = image[..., :3]  # 仅保留RGB通道
        image = torch.from_numpy(image).permute(2, 0, 1)  # 转换为CHW格式

        # 处理文本描述
        description_vectors = text_to_vectors(description, self.word_vectors)

        return {'image': image, 'description_vectors': description_vectors, 'description': description}


def collate_fn(batch):
    images = [item['image'] for item in batch]
    description = [item['description'] for item in batch]
    descriptions = [torch.tensor(item['description_vectors'], dtype=torch.float32) for item in batch]
    descriptions_padded = pad_sequence(descriptions, batch_first=True, padding_value=0)
    images = torch.stack(images, dim=0)  # 现在images是Tensor列表
    return {'image': images, 'description_vectors': descriptions_padded, 'description': description}

