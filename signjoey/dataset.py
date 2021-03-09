# coding: utf-8
"""
Data module
"""
from torchtext import data
import argparse
import numpy as np
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import pickle5
import gzip
import torch
import lzma
from augmentations import load_augment
import math

def load_dataset_file(filename):
    print(f"Loading {filename}")
    with open(filename, "rb") as f:
        try:
            return pickle.load(f)
        except ValueError:
            return pickle5.load(f)


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]


        samples = {}
        for annotation_file, annotation_type in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                # Decompress and augment
                s["sign"] = load_augment(annotation_type,pickle.loads(lzma.decompress(s['sign'])))

                if seq_id in samples:
                    if samples[seq_id]["sign"].shape[0] > s["sign"].shape[0]:# If there are less symbols then pad with zero filled
                        feature_size = s["sign"].shape[1]
                        difference =  samples[seq_id]["sign"].shape[0] - s["sign"].shape[0]
                        s["sign"] = torch.cat([
                            torch.zeros(math.ceil(difference/2),feature_size),
                            s["sign"],
                            torch.zeros(math.floor(difference / 2), feature_size)
                        ],axis=0)

                    samples[seq_id]["loaded"] += 1
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                        "loaded": 1,
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            if sample['loaded'] == len(path):
                examples.append(
                    data.Example.fromlist(
                        [
                            sample["name"],
                            sample["signer"],
                            # This is for numerical stability
                            sample["sign"] + 1e-8,
                            str(sample["gloss"]).strip(),
                            str(sample["text"]).strip(),
                        ],
                        fields,
                    )
                )
            else:
                print(f"{s} only loaded {sample['loaded']} annotations so has been removed")
        super().__init__(examples, fields, **kwargs)

if __name__ == "__main__":#Get stats


    import cv2

    x_indexes = np.array([i * 3 for i in range(135)])
    y_indexes = np.array([(i * 3) + 1 for i in range(135)])

    def vis(keypoints):
        if len(keypoints.shape) > 1:
            keypoints = keypoints[0]



    parser = argparse.ArgumentParser("test-data")
    parser.add_argument(
        "--path",
        default="/mnt/vol/research/SignTranslation/data/ChaLearn2021/train/ChaLearn2021.train.openpose.fp32.slt.ui01.valid",
        type=str,
        help="Dataset path",
        required=False
    )
    args = parser.parse_args()
    path = args.path
    tmp = load_dataset_file(path)


    max_x = 0
    min_x = np.inf
    max_y = 0
    min_y = np.inf

    means = np.zeros(len(tmp))
    lengths = np.zeros(len(tmp))
    for i, s in enumerate(tmp):
        values = pickle.loads(lzma.decompress(s['sign']))#["shoulder centre"]

        xs = values[:,x_indexes]
        ys = values[:,y_indexes]
        max_x = max(max_x, xs.max())
        min_x = min(min_x, xs.min())
        max_y = max(max_y, ys.max())
        min_y = min(min_y, ys.min())

        vid_means = values.mean(axis=0)

        means[i] = (vid_means[5*3] + vid_means[6*3])/2

        lengths[i] = values.shape[0]

    print(means.shape)

    print(f"X: {max_x}   {min_x} Y: {max_y}    {min_y}")
    print(f"Shoulders have means: {means.mean()}, var: {means.std(ddof=1)}, max: {means.max()}, min: {means.min()}")
    print(f"Video length has mean: {lengths.mean()}, var: {lengths.std(ddof=1)}, max: {lengths.max()}, min: {lengths.min()}")

