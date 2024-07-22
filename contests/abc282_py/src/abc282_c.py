import sys
import math

N = int(input())
S = list(input())
isin = True
for i in range(N):
    isin ^= S[i]=='"'
    if isin and S[i]==",":
        S[i] = "."
print("".join(S))
