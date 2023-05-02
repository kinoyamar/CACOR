description = """
Run an experiment multiple times with the same config file.
"""

import os
import sys
import traceback
import argparse
import yaml
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument(
        'config_path', type=str,
        help='Path to the config file.'
    )
    parser.add_argument(
        'start_index', type=int,
        help='Start index of the run_name_suffix.'
    )
    parser.add_argument(
        'num_runs', type=int,
        help='Number of runs.'
    )
    parser.add_argument(
        '-i', '--input-size', type=int, default=None,
        help='Overrides "input_size" of the config file. Only for MNIST.'
    )
    parser.add_argument(
        '-m', '--memory-size', type=int, default=None,
        help='Overrides "rehe_patterns" (and "rehe_patterns_val") of the config file.'
    )
    parser.add_argument(
        'other_args', nargs='*', type=str,
        help='Other arguments to be passed to run_experiment.py.'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    try:
        with open(args.config_path) as fp:
            config = yaml.safe_load(fp)
    except:
        print('Error: failed to load config file.', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
    input_size = config['input_size']
    memory_size = config['rehe_patterns']
    if config['exp_type'] == 'mnist':
        if args.input_size is not None:
            input_size = args.input_size
    if args.memory_size is not None:
        memory_size = args.memory_size
    else:
        if config['method'] == 'aco':
            memory_size += config['rehe_patterns_val']
    
    run_name = config.get('run_name', 'run')
    if config['exp_type'] == 'mnist':
        run_name += f'_{input_size}'
    run_name += f'_{memory_size}'

    run_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_experiment.py')

    for i in range(args.start_index, args.start_index + args.num_runs):
        commands = [
            'sbatch',
            '--job-name', f'{run_name}_{i}',
            '--output', 'results/logs/out-%j.log',
            '~/simg/run_with_singularity.sh',
            run_script_path,
            args.config_path,
            '--run-name', run_name,
            '--run-name-suffix', f'_{i}',
            *args.other_args
        ]
        if args.input_size is not None:
            commands.extend(['--input-size', f'{input_size}'])
        if args.memory_size is not None:
            commands.extend(['--rehe-patterns', f'{memory_size}'])

        subprocess.run(
            ' '.join(commands),
            shell=True
        )

if __name__ == '__main__':
    main()
