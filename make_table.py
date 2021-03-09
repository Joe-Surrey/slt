import argparse
import yaml
import os
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.styles.borders import Border, Side
import time

horizontal_border = Border(left=Side(style=None),
                     right=Side(style=None),
                     top=Side(style='thick'),
                     bottom=Side(style=None))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


class ColCount():
    def __init__(self):
        self.alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.current = [0]

    def inc(self, index=0):
        if self.current[index] == 25:
            self.current[index] = 0
            if index == len(self.current) - 1:
                self.current.append(0)
                return
            else:
                self.inc(index + 1)
        else:
            self.current[index] += 1

    def __iadd__(self, other):
        for _ in range(other):
            self.inc()
        return self

    def __str__(self):
        return "".join([self.alph[index] for index in reversed(self.current)])


punctuation = '!?,:;"\')(_-'
def clean(s):
    return s.strip().translate(str.maketrans('', '', punctuation))


def add_row(sheet, row, data, bold_index=None, do_border=False):
    col = ColCount()
    for index, item in enumerate(data):
        cell = str(col) + str(row)
        sheet[cell] = item
        if bold_index is not None and bold_index == index:
            sheet[cell].font = Font(bold=True)
        if do_border:
            sheet[cell].border = horizontal_border
        col += 1

def add(sheet, row, cfgs,results, bold_index=None, do_test = False):
    prev_mod = None
    for name, cfg in cfgs.items():
        row += 1
        mod, seed = name.split("_")


        do_border = mod != prev_mod
        prev_mod = mod

        if results[mod][seed] is None:
            continue
        dev = results[mod][seed]["dev"]

        opt = cfg["training"]["optimizer"]
        lr = cfg["training"]["learning_rate"]
        bs = cfg["training"]["batch_size"]
        drop = cfg["model"]["encoder"]["dropout"]
        layers = cfg["model"]["encoder"]["num_layers"]
        hs = cfg["model"]["encoder"]["hidden_size"]
        heads = cfg["model"]["encoder"]["num_heads"]
        norm = cfg["model"]["encoder"]["embeddings"]["norm_type"]
        act = cfg["model"]["encoder"]["embeddings"]["activation_type"]
        scale = cfg["model"]["encoder"]["embeddings"]["scale"]
        scale = "-" if not scale else scale

        current_row = [seed, opt, lr, bs, drop, layers, hs, heads, norm, act, scale] + dev
        if do_test:
            test = results[mod][seed]["test"]
            current_row.extend(test)
        add_row(sheet, row, current_row, bold_index=bold_index, do_border=do_border)
    return row

def load(base_path,do_test = False):
    #load yamls in path
    dir_path = Path(base_path)
    print(dir_path)
    cfgs = {file_path.stem: load_config(str(file_path)) for file_path in sorted((dir_path/"configs").glob("*.yaml"))}
    print(len(cfgs))
    result_paths = [file_path for file_path in sorted(dir_path.glob("*/")) if file_path.stem not in ["logs","configs"]]

    results = {}

    for result_path in result_paths:
        #print(result_path.name)
        results[result_path.name] = {}
        seed_paths = [file_path for file_path in sorted(result_path.glob("*/"))]
        for seed in seed_paths:
            results[result_path.name][seed.stem] = {}
            try:
                with open(str(seed/"train.log")) as f:
                    log = f.read()
                if do_test:
                    log = log.split("Training ended after")[-1].split(
                        "************************************************************")
                    dev_part = [clean(item.split(" ")[-1]) for item in log[-3].split("\t")[3:8]]
                    test_part = [clean(item.split(" ")[-1]) for item in log[-2].split("\t")[3:8]]
                    results[result_path.name][seed.stem]["dev"] = dev_part
                    results[result_path.name][seed.stem]["test"] = test_part
                else:
                    log = log.split("Hooray! New best validation result")[-1]
                    log = log.split("Logging")[0].split("\t")[8:13]
                    dev_part = [clean(item.split(" ")[-1]) for item in log]
                    results[result_path.name][seed.stem]["dev"] = dev_part
            except:
                print(f"Failed to open{str(seed/'train.log')}")
                results[result_path.name][seed.stem] = None
    return cfgs,results


def do_one(args):

    base_path = "/vol/research/SignRecognition/slt/experiments/" + args.root
    #os.chdir("/vol/research/SignRecognition/slt/experiments")


    cfgs,results = load(base_path,args.test)

    base_line_configs, base_line_results = load("/vol/research/SignRecognition/slt/experiments/baseline")

    workbook = Workbook()
    sheet = workbook.active
    row = 1
    headers = ["seed","optimizer", "learning_rate", "batch_size", "dropout", "num_layers",
                   "hidden_size", "num_heads", "norm_type", "activation_type", "scale",
                   'dev WER', 'DEL', 'INS', 'SUB', 'Acc']
    test_headers = ['test WER', 'DEL', 'INS', 'SUB', 'Acc']

    if args.test:
        headers.extend(test_headers)

    if args.root in headers:
        bold_index = headers.index(args.root)
    else:
        bold_index = None


    add_row(sheet,row,headers,bold_index)
    row = add(sheet,row,cfgs,results, bold_index,args.test)

    row += 1
    row = add(sheet, row, base_line_configs, base_line_results, bold_index, args.test)

    workbook.save(filename=base_path + "/results.xlsx")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Root directory ")
    parser.add_argument("--test", type=bool, default=False, help="If testng was done ")
    args = parser.parse_args()

    if args.root is not None:
        do_one(args)


