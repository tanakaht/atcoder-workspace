import sys
import math


N, M = map(int, input().split())
# xこの要素を持つsを作る
def construct(x):
    ret = [0]
    for i in range(len(bin(x))):
        # 2倍にする
        ret2 = [x+2*ret[-1]+1 for x in ret]
        ret += ret2
    ret = ret[:x]
    return ret
S = construct(N-1)
S.append(2*S[-1]+1+2*N)
S[-1] += (M-sum(S))%N
diff = (M-sum(S))//N
ans = [s+diff for s in S]
print(*ans)
if ans[0]<-10000000 or ans[-1]>10000000:
    raise ValueError
