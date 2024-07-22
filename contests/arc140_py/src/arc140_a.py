from collections import defaultdict
from email.policy import default
import sys
import math
from collections import defaultdict

N, K = map(int, input().split())
S = input()

ans = N
for i in range(1, N+1):
    if N%i!=0:
        continue
    cnt = 0
    counter = [defaultdict(int) for _ in range(N//i)]
    for j in range(N):
        counter[j%(N//i)][S[j]] += 1
    for c in counter:
        cnt += i-max(c.values())
    if cnt <= K:
        ans = min(ans, N//i)
print(ans)
