import sys
import math

sys.setrecursionlimit(1000000)
# chain decomposition の構築
def construct(G, N):
    # P[v]: DFS-treeにおける頂点vの親頂点
    P = [0]*N
    # G0: グラフGのbackedgeのみに絞った有向グラフ
    G0 = [[] for i in range(N)]
    # V: 頂点をpre-orderに並べたもの
    V = []

    # P, V, G0 を計算するDFS
    lb = [0]*N
    def dfs(v, p):
        P[v] = p
        V.append(v)
        lb[v] = len(V)
        for w in G[v]:
            if w == p:
                continue
            if lb[w]:
                if lb[v] < lb[w]:
                    # (v, w) は backedge
                    G0[v].append(w)
                continue
            # (v, w) は tree edge
            dfs(w, v)
    dfs(0, -1)

    # B: 橋となる辺e = (u, v) のリスト
    B = []
    # ap[v]: 頂点vが関節点であるか?
    ap = [0]*N

    used = [0]*N
    first = 1
    used[0] = 1
    # グラフの頂点をpre-orderで見ていく
    for u in V:
        if not used[u]:
            # 頂点uは以前に探索されてない
            # -> この親頂点pへのtree edgeがchainとして含まれない
            # -> 辺(u, p) は橋
            p = P[u]
            B.append((u, p) if u < p else (p, u))
            # 橋に隣接し、次数が2以上の頂点は関節点
            if len(G[u]) > 1:
                ap[u] = 1
            if len(G[p]) > 1:
                ap[p] = 1

        # 頂点vが始点となるbackedgeについて調べる
        cycle = 0
        for v in G0[u]:
            # tree edgeに従って根頂点に向かって上がっていく
            w = v
            while w != u and not used[w]:
                used[w] = 1
                w = P[w]
            # このchainはcycle
            if w == u:
                cycle = 1

        if cycle:
            if not first:
                # 2つ目以降のcycle chainである場合、頂点uは関節点
                ap[u] = 1
            first = 0

    A = [v for v in range(N) if ap[v]]
    return B, A


T = int(input())
for caseid in range(1, T+1):
    N, L = map(int, input().split())
    K = []
    for _ in range(L):
        input()
        K.append(set(map(int, input().split())))
    g = [[] for _ in range(N+L)]
    for i, S in enumerate(K):
        for s in S:
            s -= 1
            g[s].append(i+N)
            g[i+N].append(s)
    ans = 0
    for x in construct(g, len(g))[1]:
        ans += x>=N
    print(f"Case #{caseid}: {ans}")
