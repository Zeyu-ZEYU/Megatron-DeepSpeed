import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from matplotlib.pyplot import cm

with open("/home/zeyu/large_files/attn_scores.pkl", "rb") as f:
    attn_scores = pickle.load(f)


def get_score(layer, head):
    return attn_scores[layer][0, head, :, :]


def repetition_per_head(head):
    data = {}
    for ly in range(24):
        score = get_score(ly, head)
        sum = torch.sum(score, dim=1)
        arr = []
        for i, s in enumerate(sum):
            arr.append((i, s.item()))
        arr.sort(key=lambda x: x[1])
        data[ly] = arr
    for ly1 in range(24):
        r_ratios = []
        print(f"layer {ly1}")
        for ly2 in range(ly1, 24):
            s1 = data[ly1]
            s2 = data[ly2]
            ratio = 0.4
            s1 = set([i for i, _ in s1[: int(2048 * ratio)]])
            s2 = set([i for i, _ in s2[: int(2048 * ratio)]])
            same = s1.intersection(s2)
            repetition_ratio = len(same) / len(s1)
            r_ratios.append(repetition_ratio)
        r_ratios = [f"{x:.2f}" for x in r_ratios]
        print(r_ratios)


def repetition_per_layer(layer):
    data = {}
    for head in range(16):
        score = get_score(layer, head)
        sum = torch.sum(score, dim=1)
        arr = []
        for i, s in enumerate(sum):
            arr.append((i, s.item()))
        arr.sort(key=lambda x: x[1])
        data[head] = arr
    for h1 in range(16):
        r_ratios = []
        print(f"head {h1}")
        for h2 in range(h1, 16):
            s1 = data[h1]
            s2 = data[h2]
            ratio = 0.35
            s1 = set([i for i, _ in s1[: int(2048 * ratio)]])
            s2 = set([i for i, _ in s2[: int(2048 * ratio)]])
            same = s1.intersection(s2)
            repetition_ratio = len(same) / len(s1)
            r_ratios.append(repetition_ratio)
        r_ratios = [f"{x:.2f}" for x in r_ratios]
        print(r_ratios)


for head in range(24):
    repetition_per_layer(head)
