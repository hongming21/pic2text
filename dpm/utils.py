import importlib
import json
import re
def instantiate_from_config(config):

    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_vocabulary(path):
    with open(path, 'r') as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict
def convert_to_word_lists(batch):
    # 正则表达式，用于拆分单词，保留句点作为独立单词
    regex = r'\w+|[.]'
    
    new_batch = []
    for item in batch:
        # 提取字符串
        sentence = item[0]
        # 使用正则表达式拆分为单词列表
        words = re.findall(regex, sentence)
        new_batch.append(words)
    return new_batch
