import json
from pathlib import Path
import string
def create_and_save_vocab(json_file_paths, save_path='./data/vocabulary.json'):
        """
        如果不存在，则从两个描述文件中创建词汇表并将其保存到本地。

        Args:
            json_file_paths (list of Path objects): 包含图片名称和对应描述的JSON文件路径列表。
            save_path (Path): 保存词汇表的本地路径。
        """
        def preprocess_text(text):
            """
            对文本进行预处理，将句号与单词分离。
            """
            # 使用空格替换句号，确  保句号被视为独立的单词
            text = text.replace('.', ' . ')
            return text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))

        save_path = Path(save_path)
        if save_path.exists():
            print("Vocabulary file already exists. Skipping creation.")
            return

        word_freq = {}
        for json_file_path in json_file_paths:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                descriptions = json.load(file)

            for _, description in descriptions.items():
                preprocessed_text = preprocess_text(description)
                words = preprocessed_text.split()
                for word in words:
                    if word not in word_freq:
                        word_freq[word] = 1
                    else:
                        word_freq[word] += 1
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3, '.': 4}
        idx = 5
        for word in sorted(word_freq, key=word_freq.get, reverse=True):
            if word not in vocab:
                vocab[word] = idx
                idx += 1

        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(vocab, file)
            print('save vocabulary!')

if __name__=="__main__":
    train_path='/root/pic2text/data/train_captions.json'
    test_path='/root/pic2text/data/test_captions.json'
    create_and_save_vocab([train_path,test_path])
        