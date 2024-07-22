import sys
import math

N = int(input())
S = list(map(int, input()))
ans = 0
dp = [0, 0]
for i in S:
    dp0, dp1 = dp[0], dp[1]
    if i:
        dp[0] = dp1
        dp[1] = dp0+1
    else:
        dp[0] = 1
        dp[1] = dp0+dp1
    ans += dp[1]

print(ans)
