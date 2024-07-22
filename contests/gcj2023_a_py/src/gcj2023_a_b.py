import sys
import math

T = int(input())
for caseid in range(1, T+1):
    M, R, N = map(int, input().split())
    X = list(map(int, input().split()))
    left = 0
    ans = 0
    for i, x in enumerate(X):
        if i<N-1 and X[i+1]-R<=left:
            continue
        elif left<M and x-R<=left:
            ans += 1
            left = x+R
        else:
            break
    if left < M:
        ans = "IMPOSSIBLE"
    print(f"Case #{caseid}: {ans}")
