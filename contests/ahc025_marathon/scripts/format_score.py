import sys
import matplotlib.pyplot as plt
import math

d = {}
with open("./dev/tmp_scores.txt") as f:
    for line in f:
        try:
            i, score = map(int, line.split())
            d[i] = math.log(score)
        except ValueError as e:
            pass
# comment = sys.argv()[1]
scores = []
for i, score in sorted(d.items(), key=lambda x: x[0]):
    scores.append(score)
plt.plot(range(len(scores)), scores)
plt.savefig("./results/score.png")
print(sum(d.values())/len(d))
