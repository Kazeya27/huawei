import json
import os


def parse_config_file(config_file):
    config = {}
    if config_file is not None:
        if os.path.exists('./config/{}.json'.format(config_file)):
            with open('./config/{}.json'.format(config_file), 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in config:
                        config[key] = x[key]
        else:
            raise FileNotFoundError(
                'Config file {}.json is not found. Please ensure \
                the config file is in the root dir and is a JSON \
                file.'.format(config_file))