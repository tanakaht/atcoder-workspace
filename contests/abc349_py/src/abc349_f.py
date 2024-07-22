import sys
import math

MOD = 998244353
N, M = map(int, input().split())
A = list(map(int, input().split()))
A = [a for a in A if M%a==0]

def factorization(n):
    arr = []
    temp = n
    for i in range(2, int(-(-n**0.5//1))+1):
        if temp%i==0:
            cnt=0
            while temp%i==0:
                cnt+=1
                temp //= i
            arr.append([i, cnt])

    if temp!=1:
        arr.append([temp, 1])

    if arr==[]:
        arr.append([n, 1])
    return arr

factors = factorization(M)
factor_cnt = [0]*(1<<len(factors))
for a in A:
    idx = 0
    for i, (p, cnt) in enumerate(factors):
        if a%(pow(p, cnt))==0:
            idx += 1<<i
    factor_cnt[idx] += 1
dp = [0]*(1<<len(factors))
dp[0] = 1
pow2 = [1]
for i in range(1, max(factor_cnt)+1):
    pow2.append(pow2[-1]*2%MOD)
for i, cnt in enumerate(factor_cnt):
    for j in range(len(dp)-1, -1, -1):
        dp[j|i] = (dp[j|i]+dp[j]*(pow2[cnt]-1))%MOD
print(dp[-1])
