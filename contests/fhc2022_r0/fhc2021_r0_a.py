import sys
from collections import defaultdict

T = int(input())
for caseid in range(1, T+1):
    N, K = map(int, input().split())
    S = sorted(list(map(int, input().split())))
    cnt = defaultdict(int)
    ans = "YES" if N<=2*K else "NO"
    for s in S:
        cnt[s] += 1
        if cnt[s] > 2:
            ans = "NO"
    print(f'Case #{caseid}: {ans}')
