import yaml

class DictConfig(object):
    """Creates a Config object from a dict
       such that object attributes correspond to dict keys.
    """
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            # Flatten any list of lists
            if isinstance(val, list) and any(isinstance(i, list) for i in val):
                # Flatten the list of lists
                val = [item for sublist in val for item in sublist]
            self.__setattr__(key, val)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())


def get_config(fname):
    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader=yaml.FullLoader)
    config = DictConfig(config_dict)
    return config