import sys
import math

T = int(input())
for caseid in range(1, T+1):
    N, K = map(int, input().split())
    A_ = list(map(int, input().split()))
    A = sorted(A_)
    nexts = [None]*N
    pres = [None]*N
    right = 0
    for left in range(N):
        while right<N and A[right]-A[left]<K:
            right += 1
        if right < N:
            nexts[left] = right
    left = N-1
    for right in range(N-1, -1, -1):
        while left>=0 and A[right]-A[left]<K:
            left -= 1
        if left>=0:
            pres[right] = left
    dp_left = [0]*N
    dp_right = [0]*N
    for i in range(N):
        if pres[i] is None:
            dp_left[i] = 1
        else:
            dp_left[i] = 1+dp_left[pres[i]]
    for i in range(N-1, -1, -1):
        if nexts[i] is None:
            dp_right[i] = 1
        else:
            dp_right[i] = 1+dp_right[nexts[i]]
    anss = {}
    for i in range(N):
        if nexts[i] is None:
            anss[A[i]] = dp_left[i]
        else:
            anss[A[i]] = dp_left[i] + dp_right[nexts[i]]
    anss_ = []
    for a in A_:
        anss_.append(anss[a])
    print(f"Case #{caseid}: {' '.join(map(str, anss_))}")
