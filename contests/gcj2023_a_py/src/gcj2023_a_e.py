import sys
import math


T = int(input())
for caseid in range(1, T+1):
    S = list(input())
    ans = math.inf
    for i in range(len(S)):
        tmpans = 0
        pre = None
        for x in range(i, i+len(S)):
            x = x%len(S)
            if pre==S[x]:
                tmpans += 1
                pre = None
            else:
                pre = S[x]
        if pre == S[i]:
            tmpans += 1
        ans = min(ans, tmpans)
    print(f"Case #{caseid}: {ans}")
