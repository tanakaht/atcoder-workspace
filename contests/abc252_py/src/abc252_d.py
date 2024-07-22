import sys
import math
from collections import Counter

N = int(input())
A = Counter(map(int, input().split()))
ans = 0
cnts = 0
ptn2 = 0
for a, cnt in A.items():
    ans += cnt*ptn2
    ptn2 += cnts*cnt
    cnts += cnt
print(ans)
