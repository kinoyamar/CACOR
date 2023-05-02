description = """
Run an experiment from a config file.

Some parameters in the config file can be overwritten by optional command line arguments.
See help (-h, --help) for the full list.
"""

import sys
import argparse
import traceback

from clutils.extras import parse_config
from experiments.mnist import mnist_exp
from experiments.quickdraw import quickdraw_exp
from experiments.speech_words import speech_words_exp

def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        'config_path', type=str,
        help='Path to the config file.'
    )

    # Optional arguments
    parser.add_argument(
        '--run-name', type=str, default=None,
        help='Overrides "run_name" property of the config file.'
    )
    parser.add_argument(
        '-s', '--run-name-suffix', type=str, default=None,
        help='"run_name" will be suffixed with this.'
    )
    parser.add_argument(
        '--input-size', type=int, default=None,
        help='Overrides "input_size" and "pixel_in_input" properties of the config file. '
        'Only valid for MNIST experiments.'
    )
    parser.add_argument(
        '--rehe-patterns', type=int, default=None,
        help='Overrides "rehe_patterns" property of the config file. '
        'For the method "aco", this will also affect "rehe_patterns_val. '
        'See also "--aco-memory-split".'
    )
    parser.add_argument(
        '--aco-memory-split', type=float, default=0.2,
        help='Only valid for the method "aco". '
        '"rehe_patterns_val" will be set to "--rehe-patterns" * "--aco-memory-split", '
        'and "rehe_patterns" will be "--rehe-patterns" - "rehe_patterns_val". '
        'This will only take effect when "--rehe-patterns" is specified.'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    try:
        config = parse_config(args.config_path)
    except:
        print('Error: failed to load config file.\n', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Check method and experiment types are valid.
    valid_methods = [
        'replay', 'aco'
    ]
    valid_exp_types = [
        'mnist', 'quickdraw', 'speech'
    ]
    if config.method not in valid_methods:
        print('Error: invalid "method" in config.\n', file=sys.stderr)
        sys.exit(1)
    if config.exp_type not in valid_exp_types:
        print('Error: invalid "exp_type" in config.\n', file=sys.stderr)
        sys.exit(1)

    # Overriding config.
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.run_name_suffix is not None:
        config.run_name += args.run_name_suffix
    if args.input_size is not None:
        config.input_size = args.input_size
        config.pixel_in_input = args.input_size
    if args.rehe_patterns is not None:
        if config.method == 'aco':
            config.rehe_patterns_val = int(args.rehe_patterns * args.aco_memory_split)
            config.rehe_patterns = args.rehe_patterns - config.rehe_patterns_val
        else:
            config.rehe_patterns = args.rehe_patterns
    
    # Run the experiment.
    is_aco = (config.method == 'aco')
    if config.exp_type == 'mnist':
        mnist_exp(config, is_aco=is_aco)
    elif config.exp_type == 'quickdraw':
        quickdraw_exp(config, is_aco=is_aco)
    elif config.exp_type == 'speech':
        speech_words_exp(config, is_aco=is_aco)

if __name__ == '__main__':
    main()
