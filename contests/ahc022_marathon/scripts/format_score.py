import sys
import matplotlib.pyplot as plt
import sys
d = {}
for i in range(int(sys.argv[1])+1):
    with open(f"./tools/err/{i:04}.txt", "r") as f:
        d[i] = {}
        for line in f:
            if line.startswith("Score ="):
                d[i]["score"] = int(line.split()[-1])
            elif line.startswith("Number of wrong answers ="):
                d[i]["n_wrong"] = int(line.split()[-1])
            elif line.startswith("Placement cost ="):
                d[i]["P_cost"] = int(line.split()[-1])
            elif line.startswith("Measurement cost ="):
                d[i]["M_cost"] = int(line.split()[-1])
            elif line.startswith("Measurement count ="):
                d[i]["M_cnt"] = int(line.split()[-1])
scores = [d[i]["score"] for i in range(len(d))]
plt.plot(range(len(scores)), scores)
plt.savefig("./results/score.png")
print(sum(scores)/len(scores))
