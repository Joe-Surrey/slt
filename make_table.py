import argparse
import yaml
import os
from pathlib import Path
def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

punctuation='!?,:;"\')(_-'
def clean(s):
    return s.strip().translate(str.maketrans('', '', punctuation))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Root directory ")
    args = parser.parse_args()

    base_path = "/vol/research/SignRecognition/slt/experiments/" + args.root
    #os.chdir("/vol/research/SignRecognition/slt/experiments")


    #load yamls in path
    dir_path = Path(base_path)
    print(dir_path)
    cfgs = {file_path.stem: load_config(str(file_path)) for file_path in sorted((dir_path/"configs").glob("*.yaml"))}
    print(len(cfgs))
    result_paths = [file_path for file_path in sorted(dir_path.glob("*/")) if file_path.stem not in ["logs","configs"]]

    results = {}
    for result_path in result_paths:
        results[result_path.stem] = {}
        seed_paths = [file_path for file_path in sorted(result_path.glob("*/"))]
        for seed in seed_paths:
            results[result_path.stem][seed.stem] = {}
            try:
                with open(str(seed/"train.log")) as f:
                    log = f.read().split("Training ended after")[-1].split("************************************************************")
                dev_part = [clean(item.split(" ")[-1]) for item in log[-3].split("\t")[3:8]]
                test_part = [clean(item.split(" ")[-1]) for item in log[-2].split("\t")[3:8]]
                results[result_path.stem][seed.stem]["dev"] = dev_part
                results[result_path.stem][seed.stem]["test"] = test_part
            except:
                results[result_path.stem][seed.stem] = None

    print(f"seed\topt\tlr\tbs\tdrop\tlayers\ths\theads\tnorm\tact\tscale\tdev WER\tDEL\tINS\tSUB\tAcc\ttest WER\tDEL\tINS\tSUB\tAcc")
    for name,cfg in cfgs.items():
        mod, seed = name.split("_")

        if results[mod][seed] is None:
            continue
        dev = '\t'.join(results[mod][seed]["dev"])
        test = '\t'.join(results[mod][seed]["test"])

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
        scale = "-" if  not scale else scale
        print(f"{seed}\t{opt}\t{lr}\t{bs}\t{drop}\t{layers}\t{hs}\t{heads}\t{norm}\t{act}\t{scale}\t{dev}\t{test}")
