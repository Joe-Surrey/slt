import argparse
import yaml
#from signjoey.helpers import load_config, save_config
import copy
import os


def modify(node,mod_key,mod_value):
    node_count = 0
    for key, item in node.items():
        if key == mod_key:
            node[key] = mod_value
            node_count += 1
        elif isinstance(item,dict):
            sub_node_count = modify(item,mod_key,mod_value)
            #if sub_node_count > 0:
                #print(f"{key}: {sub_node_count} instances changed")
            node_count += sub_node_count
    return node_count

def parse(value):
    if isinstance(value,str):
        if value.isnumeric():
            if "." in value:
                return float(value)
            else:
                return int(value)
    return value

def dirify(out_dir):
    if not out_dir.endswith("/"):
        out_dir += "/"
    return out_dir
def mkdir(out_dir):
    #out_dir = dirify(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        print(f"Warning: {out_dir} already exists")
def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def save_config(path,cfg) -> None:
    """
    Saves a config file to a YAML
    """
    if not path.endswith(".yaml"):
        path += ".yaml"
    with open(path, 'w', encoding="utf-8") as ymlfile:
        yaml.dump(cfg, ymlfile, default_flow_style=False)
if __name__ == "__main__":

    os.chdir("/vol/research/SignRecognition/slt")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mod", type=str, help="Name of variable to change"
                              "optimizer\n"
                              "learning_rate\n"
                              "batch_size\n"
                              "dropout\n"
                              "num_layers\n"
                              "hidden_size\n"
                              "num_heads\n"
                              "norm_type\n"
                              "activation_type\n"
                              "scale",
    )
    parser.add_argument(
        "values",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
    )
    parser.add_argument(
        "--base_cfg",
        default="configs/sign_recognition_chalearn_openpose.yaml",
        type=str,
        help="Base config file (yaml).",
    )
    parser.add_argument(
        "--out_path",
        default="experiments/",
        type=str,
        help="Base output directory for experiment",
    )
    parser.add_argument(
        "--seeds",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[111,222,333]
    )

#parse args
    args = parser.parse_args()
    cfg_path = args.base_cfg
    mod_key = args.mod
    values = args.values
    out_path_base = args.out_path
    seeds = args.seeds

    print(cfg_path, mod_key, values, out_path_base)
#load base config
    base_cfg = load_config(cfg_path)

#make experiemnt directory
    out_dir = out_path_base + mod_key
    mkdir(out_dir)

#Make new configs
    cfgs = []
    cfg_paths = []
    for value in values:
        path = dirify(mod_key) + str(value)
        mkdir(out_path_base + path)
        for seed in seeds:
            seed_path = dirify(dirify(mod_key) + str(value)) + str(seed)
            mkdir(out_path_base +seed_path)
            cfgs.append(copy.deepcopy(base_cfg))
            value = parse(value)
            num_changes = modify(cfgs[-1], mod_key, value)  # Modify the value to be changed
            if mod_key == "hidden_size":
                num_changes += modify(cfgs[-1], "embedding_dim", value)
            print(f"Total: {num_changes} instances changed")

            cfgs[-1]["training"]["model_dir"] += dirify(seed_path)  # Modify model path
            cfg_paths.append(out_dir + f"/configs/{value}_{seed}.yaml")

    mkdir(out_dir + "/configs")
    for cfg, path in zip(cfgs,cfg_paths):
        save_config(path,cfg)
#Make submit file
    with open("train.submit_file","r") as f:
        file = f.read()
    mkdir(f"{out_dir}/logs")
    app = f"log    = /vol/research/SignRecognition/slt/{out_dir}/logs/c$(cluster).p$(process).log\n"+\
        f"output = /vol/research/SignRecognition/slt/{out_dir}/logs/c$(cluster).p$(process).out\n"+\
        f"error  = /vol/research/SignRecognition/slt/{out_dir}/logs/c$(cluster).p$(process).error\n"
    for path in cfg_paths:
        app += f"args = ../{path} --wkdir /vol/research/SignRecognition/slt/signjoey\nqueue\n"
    file += app

    with open(out_dir + "/train.submit_file","w") as f:
        f.write(file)

    for cfg in cfgs:
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
        scale = "-" if scale == "false" else scale
        print(f"{opt}\t{lr}\t{bs}\t{drop}\t{layers}\t{hs}\t{heads}\t{norm}\t{act}\t{scale}")
    print(f"condor_submit /vol/research/SignRecognition/slt/{out_dir}/train.submit_file")



#Opt.	LR	BS	Drop	Layer	HS	Heads	Normalization	Activation	Scale
