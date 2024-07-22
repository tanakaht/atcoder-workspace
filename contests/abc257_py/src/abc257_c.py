import sys
import math

N = int(input())
S = input()
W = list(map(int, input().split()))
events = []
for w, s in zip(W, S):
    if s == "0":
        events.append((w, 1))
    else:
        events.append((w, -1))
events = sorted(events)
anstmp = S.count("1")
ans = anstmp
for _, x in events:
    anstmp += x
    ans = max(ans, anstmp)
print(ans)
