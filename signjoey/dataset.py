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
import gzip
import torch
import lzma

def load_dataset_file(filename):
    print(f"Loading {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)


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

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                s["sign"] = torch.Tensor(np.delete(pickle.loads(lzma.decompress(s['sign'])), (range(13 * 3, 25 * 3)), axis=1))
                if seq_id in samples:
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
                    }

        examples = []
        for s in samples:
            sample = samples[s]
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
        super().__init__(examples, fields, **kwargs)

if __name__ == "__main__":#Get stats
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

    means = np.zeros(len(tmp))
    lengths = np.zeros(len(tmp))
    for i,s in enumerate(tmp):
        values = pickle.loads(lzma.decompress(s['sign']))#["shoulder centre"]
        vid_means = values.mean(axis = 0)

        means[i] = (vid_means[5*3] + vid_means[6*3])/2

        lengths[i] = values.shape[0]

    print(means.shape)
    print(f"Shoulders have means: {means.mean()}, var: {means.var(ddof=1)}, max: {means.max()}, min: {means.min()}")
    print(f"Video length has mean: {lengths.mean()}, var: {lengths.var(ddof=1)}, max: {lengths.max()}, min: {lengths.min()}")

