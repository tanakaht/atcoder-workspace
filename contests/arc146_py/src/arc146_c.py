from pkgutil import ModuleInfo
import sys
import math

N = int(input())
MOD = 998244353
ans = 0
rests = 1
for i in range(N+1):
    new_rest = pow(2, pow(2, i)-i-1, MOD)
    tmp = new_rest*pow(rests, MOD-2, MOD)
    if new_rest!=0:
        ans = (ans+(tmp)*pow(2, N-i, MOD))%MOD
    rests += new_rest
print((pow(2, pow(2, N), MOD)-ans)%MOD)
