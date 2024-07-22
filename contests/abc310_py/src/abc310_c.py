import sys
import math

N = int(input())
S = [input() for _ in range(N)]
for i, s in enumerate(S):
    if s>s[::-1]:
        S[i] = s[::-1]
print(len(set(S)))
