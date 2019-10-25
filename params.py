import configargparse
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_params():
    p = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=['default.yaml'])

    p.add('-v', help='verbose', action='store_true')

    p.add('--n-epochs', type=int)
    p.add('--batch-size', type=int)
    p.add('--n-workers', type=int)
    p.add('--in-shape', type=int)
    p.add('--sig-x', type=float)
    p.add('--sig-y', type=float)

    return p
