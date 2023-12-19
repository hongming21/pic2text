import string
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from PIL import Image
import json
import os
import numpy as np


class ImageTextDataset(Dataset):
    def __init__(self, json_file_path, image_folder_path, embedding_dim):
        """
        Args:
            json_file_path (string): JSON文件的路径，包含图片名称和对应的描述。
            image_folder_path (string): 包含图片的文件夹路径。
            embedding_dim (int): 词向量维度。
        """
        with open(json_file_path, 'r', encoding='utf-8') as file:
            self.descriptions = json.load(file)
        self.image_folder_path = image_folder_path
        self.word_to_index = self.create_vocab(self.descriptions)
        vocab_size = len(self.word_to_index)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def create_vocab(self, descriptions):
        word_freq = {}
        for _, description in descriptions.items():
            words = description.lower().translate(str.maketrans('', '', string.punctuation)).split()
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

        # 添加特殊符号
        vocab = {'<sos>': 0, '<eos>': 1, '<unk>': 2}
        idx = 3  # 从索引3开始添加其他单词

        for word in sorted(word_freq, key=word_freq.get, reverse=True):
            vocab[word] = idx
            idx += 1

        return vocab

    def text_to_indices(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        # 添加开始符和结束符
        indices = [self.word_to_index['<sos>']] + \
                  [self.word_to_index.get(word, self.word_to_index['<unk>']) for word in text.split()] + \
                  [self.word_to_index['<eos>']]
        return indices

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        image_name, description = list(self.descriptions.items())[idx]
        image_path = os.path.join(self.image_folder_path, image_name)

        image = Image.open(image_path)
        image = image.resize((750, 1101))
        image = np.array(image)
        if image.shape[2] == 4:
            image = image[..., :3]
        image = torch.from_numpy(image).permute(2, 0, 1)

        indices = self.text_to_indices(description)
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        description_vectors = self.embedding(indices_tensor)

        return {'image': image, 'description_vectors': description_vectors, 'description': description}


def collate_fn(batch):
    images = [item['image'] for item in batch]
    descriptions = [item['description_vectors'] for item in batch]
    description_word = [item['description'] for item in batch]
    descriptions_padded = pad_sequence(descriptions, batch_first=True, padding_value=0)
    images = torch.stack(images, dim=0)
    return {'image': images, 'description_vectors': descriptions_padded, 'description': description_word}
