from email.policy import default
import math
import copy

class Mo:
    def __init__(self, N, init_state, add, remove, state2ans, block_len=None):
        """
        input:
            N: 区間の長さ
            init_state: [0, 0)でのstate
            add: (i, state)->state: iの要素を追加したstateをreturn
            remove: (i, state)->state: iの要素を削除したstateをreturn
            state2ans: stateから解答
            block_len: blockの長さ
        """
        self.n = N
        self.init_state = init_state
        self.add = add
        self.remove = remove
        self.state2ans = state2ans
        self.block_len = block_len or int(math.sqrt(self.n))

    def solve(self, queries):
        """
        input:
            queries: List of (l, r)
        return:
            anss: List of [li, ri)についての解答
        """
        anss = [None]*len(queries)
        block_cnt = self.n//self.block_len
        Qs = [[] for _ in range(block_cnt+2)]
        for i, (l, r) in enumerate(queries):
            idx = l//self.block_len
            Qs[idx].append((l, r, i))
        for i in range(block_cnt+2):
            Qs[i] = sorted(Qs[i], key=lambda x: x[1])[::(1-2*(i%2))]
        curl, curr = 0, 0
        state = copy.deepcopy(self.init_state)
        for Q in Qs:
            for l, r, i in Q:
                # 拡張
                while curl > l:
                    curl -= 1
                    state = self.add(curl, state)
                while curr < r:
                    curr += 1
                    state = self.add(curr-1, state)
                while curl < l:
                    curl += 1
                    state = self.remove(curl-1, state)
                while curr > r:
                    curr -= 1
                    state = self.remove(curr, state)
                anss[i] = self.state2ans(state)
        return anss


import sys
from collections import defaultdict


N = int(input())
A = list(map(int, input().split()))
Q = int(input())
Qs = [list(map(int, input().split())) for _ in range(Q)]
queries = []
for l, r in Qs:
    queries.append((l-1, r))
init_state = [defaultdict(int), 0]
def state2ans(state):
    return state[1]
def add(i, state):
    a = A[i]
    state[0][a] += 1
    state[1] += (state[0][a]%2==0)
    return state
def remove(i, state):
    a = A[i]
    state[0][a] -= 1
    state[1] -= (state[0][a]%2==1)
    return state
mo = Mo(N, init_state, add, remove, state2ans, block_len=N//int(math.sqrt(Q)))
anss = mo.solve(queries)
print(*anss, sep='\n')
