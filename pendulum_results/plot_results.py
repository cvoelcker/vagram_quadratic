import csv
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import matplotlib as mpl

sns.set()
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.family"] = "Helvetica"
mpl.rcParams["axes.grid"] = True


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
MAX = 150

mse_files = [f"tmp/mse_{i}.txt" for i in range(6)]
vagram_files = [f"tmp/vagram_{i}.txt" for i in range(6)]
vagram_full_files = [f"tmp/vagram_full_{i}.txt" for i in range(6)]

for i, (file_names, label) in enumerate(
    zip(
        [mse_files, vagram_files, vagram_full_files],
        [
            "MSE",
            "VaGraM (1st order)",
            "VaGraM (2nd order)",
        ],
    )
):
    rews = []
    for file_name in file_names:
        rew = []
        steps = []
        with open(file_name, "r") as f:
            reader = csv.DictReader(f, delimiter=" ", fieldnames=["steps", "reward"])
            for line in reader:
                rew.append(float(line["reward"]))
                steps.append(int(line["steps"]))
        rews.append(np.array(rew)[:MAX])
        print(len(rew))
    steps = np.array(steps)[:MAX]
    rews = np.array(rews)
    m = rews.mean(0)
    s = rews.std(0)

    plt.plot(steps, m, color=colors[i], label=label)
    plt.fill_between(steps, m - s, m + s, color=colors[i], alpha=0.1)
plt.title("Pendulum-v1")
plt.ylabel("Cummulative reward")
plt.xlabel("Environment steps")
plt.legend()
plt.show()
