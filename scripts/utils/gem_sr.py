import sys
import json
import os
import numpy as np

data = []
successes = []
base = sys.argv[1]

for filename in os.listdir(base):
    if filename.endswith(".jsonl"):
        with open(os.path.join(base, filename), "r") as f:
            for line in f.readlines():
                datum = json.loads(line)
                data.append(data)
                R = sum(datum["rewards"]) - sum(
                    x.get("aux_reward", 0) for x in datum["messages"]
                )
                successes.append(np.abs(R - 1) < 1e-4)
                # successes.append("Congratulations!" in datum["messages"][-1]["content"])

print(np.mean(successes))
