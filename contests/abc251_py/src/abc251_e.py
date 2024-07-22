import sys
import math

N = int(input())
A = list(map(int, input().split()))
dp1 = [[math.inf]*2 for _ in range(N+1)] #[0, i)まで達成済み, iが餌ある→最小
dp1[0][0] = 0
for i in range(N):
    dp1[i+1][0] = min(dp1[i+1][0], dp1[i][1])
    dp1[i+1][1] = min(dp1[i+1][1], dp1[i][1]+A[i], dp1[i][0]+A[i])
ans1 = min(dp1[-1])

dp2 = [[math.inf]*2 for _ in range(N+1)] #[0, i)まで達成済み, iが餌ある→最小
dp2[0][1] = A[-1]
for i in range(N):
    dp2[i+1][0] = min(dp2[i+1][0], dp2[i][1])
    dp2[i+1][1] = min(dp2[i+1][1], dp2[i][1]+A[i], dp2[i][0]+A[i])
ans2 = min(dp2[-2])

print(min(ans1, ans2))
