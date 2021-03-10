import argparse
import yaml
#from signjoey.helpers import load_config, save_config
import copy
import os


config_files = {
    "openpose": "configs/sign_recognition_chalearn_openpose.yaml",
    "op":       "configs/sign_recognition_chalearn_openpose.yaml",
    "eff":      "configs/sign_recognition_chalearn_eff.yaml",
    "op_eff":   "configs/sign_recognition_chalearn_op_eff.yaml",
    "i3d":      "configs/sign_recognition_chalearn_i3d.yaml",
    "op_i3d":   "configs/sign_recognition_chalearn_op_i3d.yaml",
    "op_holistic": "configs/sign_recognition_chalearn_op_holistic.yaml",
    "holistic": "configs/sign_recognition_chalearn_holistic.yaml",
}


def out(cfg):
    scale = cfg["model"]["encoder"]["embeddings"]["scale"]
    vals = {
    "opt" : cfg["training"]["optimizer"],
    "lr" : cfg["training"]["learning_rate"],
    "bs" : cfg["training"]["batch_size"],
    "drop" : cfg["model"]["encoder"]["dropout"],
    "layers" : cfg["model"]["encoder"]["num_layers"],
    "hs" : cfg["model"]["encoder"]["hidden_size"],
    "heads" : cfg["model"]["encoder"]["num_heads"],
    "norm" : cfg["model"]["encoder"]["embeddings"]["norm_type"],
    "act" : cfg["model"]["encoder"]["embeddings"]["activation_type"],
    "scale" : "-" if not scale else scale,
    }

    return "\t".join(": ".join(str(n) for n in item) for item in vals.items())

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=str
    )
    parser.add_argument(
        "--scheduling",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=str
    )
    parser.add_argument(
        "--learning_rate",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=float
    )
    parser.add_argument(
        "--batch_size",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=int
    )
    parser.add_argument(
        "--dropout",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=float
    )
    parser.add_argument(
        "--hidden_size",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=int
    )
    parser.add_argument(
        "--num_heads",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
    )
    parser.add_argument(
        "--norm_type",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=str
    )
    parser.add_argument(
        "--activation_type",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=str
    )
    parser.add_argument(
        "--scale",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=bool
    )

    parser.add_argument(
        "name",  #
        type=str,
    )
    parser.add_argument(
        "--base_cfg",
        default="openpose",
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
        "--type",
        default="openpose",
        type=str,
        help="Base output directory for experiment",
    )
    parser.add_argument(
        "--comb",
        default="combine",
        type=str,
        help="How to combine if there are multiple modifications: product, combine, seperate",
    )
    parser.add_argument(
        "--seeds",  #
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[111, 222, 333]
    )

    return parser

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
    if isinstance(value,bool):
        return value
    try:
        value = int(value)
    except:
        try:
            value = float(value)
        except:
            pass
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
#parse args
    parser = args()

    params = parser.parse_args()

    ignore_args = ["name","base_cfg","out_path","type","comb","seeds",]

    args = {key: value for key, value in vars(params).items() if key not in ignore_args and value is not None}

    name = params.name
    cfg_path = config_files[params.base_cfg]
    out_path_base = params.out_path
    seeds = params.seeds
    data_type = params.type


    if params.comb == "combine":
        values =  zip(*[[(key,item) for item in items] for key, items in args.items()])
    elif params.comb == "seperate":
        values = ((key,item) for key,items in args.items() for item in items)
    else:#product
        import itertools
        values = itertools.product(*[[(key,item) for item in items] for key, items in args.items()])

    #todo add combine and seperate
    #print(list(values))
#load base config
    base_cfg = load_config(cfg_path)

#make experiemnt directory
    out_dir = out_path_base + name
    mkdir(out_dir)

#Make new configs
    cfgs = []
    cfg_paths = []
    indexes = []
    for index, value in enumerate(values):
        mod_value = parse(value)
        path = dirify(name) + str(index)
        mkdir(out_path_base + path)
        for seed in seeds:
            seed_path = dirify(dirify(name) + str(index)) + str(seed)
            mkdir(out_path_base + seed_path)
            cfgs.append(copy.deepcopy(base_cfg))
            for mod_key,mod_value in value:
                num_changes = modify(cfgs[-1], mod_key, mod_value)  # Modify the value to be changed
                if mod_key == "hidden_size":
                    num_changes += modify(cfgs[-1], "embedding_dim", mod_value)
                print(f"{mod_key} => {mod_value}. {num_changes} instances changed")

            cfgs[-1]["training"]["model_dir"] += dirify(seed_path)  # Modify model path
            cfgs[-1]["training"]["random_seed"] = seed

            if data_type is not None:
                cfgs[-1]["data"]["dataset_type"] = data_type
            cfg_paths.append(out_dir + f"/configs/{index}_{seed}.yaml")
            indexes.append(index)
# save configs and Make experiment explanation file
    exp_file = ""
    mkdir(out_dir + "/configs")
    for index, cfg, path in zip(indexes, cfgs, cfg_paths):
        save_config(path, cfg)
        exp_file += str(index) + "\t" + out(cfg) + "\n"

    with open(out_dir + "/exp.txt","w") as f:
        f.write(exp_file)


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
