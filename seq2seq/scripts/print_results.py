
import json
import sys
import numpy as np

tasks = ["cola", "mrpc", "qnli", "rte", "sst2", "mnli", "qqp", "stsb"]

file_prefix = sys.argv[1]

print(file_prefix)

for task in tasks:
    metrics = []
    for seed in [0]:
        file_name = f"{file_prefix}{task}@{seed}.json"

        data = json.load(open(file_name, "r"))

        # print(data.keys())
        metric = data["test_average_metrics"]

        metrics.append(metric)

    print(f"[{task}] avg: {np.mean(metrics):.2f} std: {np.std(metrics):.2f}")

avg_metrics = []

for seed in [0, 1, 2]:
    metrics = []
    for task in tasks:
        file_name = f"{file_prefix}{task}@{seed}.json"

        data = json.load(open(file_name, "r"))

        # print(data.keys())
        metric = data["test_average_metrics"]

        metrics.append(metric)

    avg_metrics.append(np.mean(metrics))

print(avg_metrics)

print(f"mean: {np.mean(avg_metrics):.2f} std: {np.std(avg_metrics):.2f}")
