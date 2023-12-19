import importlib
import json
def instantiate_from_config(config):

    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))


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