import sys
import math

N, M = map(int, input().split())
A = list(map(int, input().split()))
d = {i+1: 0 for i in range(N)}
max_i = 1
max_v = 0
for a in A:
    d[a] += 1
    if d[a] > max_v or (d[a]==max_v and a<max_i):
        max_i = a
        max_v = d[a]
    print(max_i)

