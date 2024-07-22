import sys
import math

N = int(input())
A = list(map(int, input().split()))
odds = sorted([a for a in A if a%2==1])
evans = sorted([a for a in A if a%2==0])
ans = -1
if len(odds)>=2:
    ans = max(ans, sum(odds[-2:]))
if len(evans)>=2:
    ans = max(ans, sum(evans[-2:]))
print(ans)
