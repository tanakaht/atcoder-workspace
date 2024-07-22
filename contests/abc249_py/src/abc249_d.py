import sys
import math
from collections import Counter

N = int(input())
A = list(map(int, input().split()))
Ac = Counter(A)
ans = 0
maxA = max(A)
for i in range(1, maxA+1):
    for j in range(1, maxA//i+1):
        ans += Ac[i]*Ac[j]*Ac[i*j]
print(ans)
