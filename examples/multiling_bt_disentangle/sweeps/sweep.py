
from sys import stdin, stdout
from typing import List
import numpy as np
import os
import argparse
from copy import deepcopy
import subprocess
from subprocess import Popen, PIPE
from shutil import copyfile

GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "")
N_GPUS = len(GPUS.split(","))

class HyperParam(object):
    def __init__(self, parse_key, arguments, save_key_fn=None, binary_flag=False) -> None:
        super().__init__()
        self.parse_key = parse_key
        self.arguments = arguments if isinstance(arguments, list) else [arguments]
        self.save_key_fn=save_key_fn
        self.binary_flag = binary_flag
        assert len(self.arguments) == 1, f'currently restrict arguments to 1 only'

    def parse_arguments(self):
        return [self.get_argument_parse(i) for i in range(len(self.arguments))]

    def get_argument_parse(self, index):
        arg = self.arguments[index]
        if self.binary_flag:
            return [self.parse_key] if arg else []
        else:
            return [self.parse_key, arg]
    
    def get_key_name(self, index):
        if self.save_key_fn is not None:
            key_name = self.save_key_fn(self.arguments[index])
            return key_name
        else:
            return ""


def get_args(add_extra_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-command', default='fairseq-train', type=str)
    parser.add_argument('--data', default=None, type=str)
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--save-prefix', default=None, type=str)
    parser.add_argument('--last-checkpoint', default=None, type=str, help="save last checkpoint into save_dir")
    parser.add_argument('--no-log-file', action="store_true", default=False)
    parser.add_argument('--no-save-dir', action="store_true", default=False)
    
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--target_lang', type=str, default="ne_NP")
    parser.add_argument('--to_english', action="store_true", default=False)
    if callable(add_extra_args):
        add_extra_args(parser)
    args = parser.parse_args()
    return args


def launch_subprocess(comamnds: List[str]):
    process = Popen(comamnds, stdout=None, stdin=None)
    print("Process running: {} / pid={}".format(process, process.pid))
    print('-' * 100)
    return process


def launch_recursive_grid(args, commands: List, grid: List[HyperParam], processes=None, series_key_name=""):
    if len(grid) == 0:
        # add save dir
        if args.save_dir is None and not args.no_save_dir:
            assert args.save_prefix is not None
            if N_GPUS > 0:
                series_key_name += f".{N_GPUS}gpu"
            save_basename = series_key_name
            args.save_dir = os.path.join(args.save_prefix, save_basename)

        # add save dir
        if not args.no_save_dir:
            commands.extend(["--save-dir", args.save_dir])
        if not args.no_log_file:
            os.makedirs(args.save_dir, exist_ok=True)
            commands.extend(["--log-file", os.path.join(args.save_dir, "train.log")])
        
        if args.last_checkpoint is not None and os.path.exists(args.last_checkpoint):
            last_ckpt = f"{args.save_dir}/checkpoint_last.pt"
            if not os.path.exists(last_ckpt):
                print(f"Copying {args.last_checkpoint} -> {last_ckpt}")
                copyfile(args.last_checkpoint, last_ckpt)
        # no more parse, start launching the command
        # final_cmd = " ".join(commands)
        commands = [str(x) for x in commands]
        print('-' * 20 + "Final Command")
        print(commands)
        print(f"Save dir: {args.save_dir}")
        print('-' * 100)

        # print("Testing----")
        # return 0
        process = launch_subprocess(commands)
        if isinstance(processes, list):
            processes.append(process)
        return process
    else:
        cur_param = grid[0]
        rest_grid = grid[1:]
        for i, arg_parse in enumerate(cur_param.parse_arguments()):
            new_commands = deepcopy(commands)
            key_name = cur_param.get_key_name(i)
            new_commands.extend(arg_parse)
            series_key_name += ("" if (series_key_name == "" or key_name == "") else ".") + key_name
            launch_recursive_grid(args, new_commands, rest_grid, processes, series_key_name=series_key_name)


def launch(args, get_grid, postprocess_hparams=None, series_key_name=""):
    if postprocess_hparams is None:
        postprocess_hparams = lambda x: x
        
    grid = get_grid(args)
    
    commands = []

    # add command
    commands.append(args.train_command)

    # add data command
    if args.data is not None:
        commands.append(args.data)
    
    processes = []
    launch_recursive_grid(args, commands, grid, processes, series_key_name=series_key_name)


def main(get_grid, postprocess_hparams, add_extra_args=None, series_key_name=""):
    args = get_args(add_extra_args=add_extra_args)
    launch(args, get_grid, postprocess_hparams, series_key_name=series_key_name)

