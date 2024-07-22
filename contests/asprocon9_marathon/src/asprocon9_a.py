import sys
import random
import math

X, M, C, E = map(int, input().split())
Costs = [[list(map(int, input().split())) for _ in range(9)] for _ in range(M)]

def cal_score(works):
    cost = 0
    for m in range(M):
        for x in range(2*X):
            cost += Costs[m][works[m][x]][x%2]
    score = round(10**9*(10-math.log(cost/X, 10)))
    return score

class State:
    def __init__(self, works):
        self.works = works

class Result:
    def __init__(self, state, score, v, d, RDRO):
        self.state = state
        self.score = score
        self.v = v
        self.d = d
        self.RDRO = RDRO
        self._diff = None
        if self.d==0:
            self.score_ignore_c = cal_score(self.state.works)
        else:
            self.score_ignore_c = 0
        self.estimate_works = [[self.state.works[m][x] if self.RDRO[m][x//2]!=0 else 0 for x in range(2*X)] for m in range(M)]

    def diff(self):
        if self._diff is not None:
            return self._diff
        res = [[0]*(2*X) for _ in range(M)]
        works = self.state.works
        for m in range(M):
            for x in range(2*X):
                rd, ro = self.RDRO[m][x//2]
                if ro>=1:
                    if random.random()<0.9:
                        res[m][x] += 1
                elif ro<0.7:
                    if ro==0 or works[m][x]!=1:
                        if random.random()<0.8:
                            res[m][x] -= 1
                elif ro>0.99 and x<2*X-2 and self.RDRO[m][x//2+1][1]>=2:
                    if random.random()<0.8:
                        res[m][x] += 1
                else:
                    if random.random()<0.1:
                        res[m][x] -= 1
                if works[m][x] == 0:
                    if random.random()<0.1:
                        res[m][x] += 1
                continue
        self._diff = res
        return res

class Results:
    def __init__(self):
        self.results = []

    def add_result(self, result):
        self.results.append(result)

    def get_new_state(self, e):
        # 早期ステージ
        if e<=E*0.25:
            return self.get_new_state_early_stage(e)
        elif E*0.25<=e<E*0.75:
            return self.get_new_state_mid_stage(e)
        elif E*0.75<=e<E-2:
            return self.get_new_state_late_stage(e)
        else:
            return self.get_new_state_final_stage(e)

    def get_new_state_early_stage(self, e):
        if len(self.results)==0:
            works = [[7]*(2*X) for _ in range(M)]
            state = State(works)
            return state
        works = [[0]*(2*X) for _ in range(M)]
        pre_work = self.results[-1].state.works
        diff = self.results[-1].diff()
        for m in range(M):
            for x in range(2*X):
                works[m][x] = min(max(0, pre_work[m][x]+diff[m][x]), 8)
        state = State(works)
        return state

    def get_new_state_mid_stage(self, e):
        return self.get_new_state_early_stage(e)
        works = [[random.randint(6, 8)]*(2*X) for _ in range(M)]
        state = State(works)
        return state

    def get_new_state_late_stage(self, e):
        max_score = 0
        for result in self.results[1:]:
            max_score = max(max_score, result.score)
        if max_score==0:
            works = [[random.randint(6, 8)]*(2*X) for _ in range(M)]
            state = State(works)
            return state
        return self.get_new_state_early_stage(e)

    def get_new_state_final_stage(self, e):
        best_works, best_score = [[8]*(2*X) for _ in range(M)], 0
        for res in self.results:
            if res.score_ignore_c==0:
                continue
            new_works = []
            costs = 0
            for m in range(M):
                # plan = res.state.works[m]
                plan = res.estimate_works[m]
                # xまで選んで、c回の切り替えを行い, 末尾が平日i, 末尾が平日j ->最小コスト
                dp = [[[[math.inf]*9 for _ in range(9)] for c in range(C+1)] for x in range(X)]
                for i in range(plan[0], 9):
                    for j in range(plan[1], 9):
                        dp[0][0][i][j] = Costs[m][i][0]+Costs[m][j][1]
                for x in range(1, X):
                    for c in range(C+1):
                        # そのまま使う
                        for i in range(plan[2*x], 9):
                            for j in range(plan[2*x+1], 9):
                                dp[x][c][i][j] = dp[x-1][c][i][j]+Costs[m][i][0]+Costs[m][j][1]
                        if c==0:
                            continue
                        # 休日だけ変更
                        for i in range(plan[2*x], 9):
                            mincost = min(dp[x-1][c-1][i])
                            for j in range(plan[2*x+1], 9):
                                dp[x][c][i][j] = min(dp[x][c][i][j], mincost+Costs[m][i][0]+Costs[m][j][1])
                        # 平日だけ変更
                        for j in range(plan[2*x+1], 9):
                            mincost = min([dp[x-1][c-1][i][j] for i in range(9)])
                            for i in range(plan[2*x], 9):
                                dp[x][c][i][j] = min(dp[x][c][i][j], mincost+Costs[m][i][0]+Costs[m][j][1])
                        if c<=1:
                            continue
                        #　両方変更
                        mincost = min([min([dp[x-1][c-2][i][j] for i in range(9)]) for j in range(9)])
                        for i in range(plan[2*x], 9):
                            for j in range(plan[2*x+1], 9):
                                dp[x][c][i][j] = min(dp[x][c][i][j], mincost+Costs[m][i][0]+Costs[m][j][1])
                # 経路復元
                mincij, mincost = None, math.inf
                for c in range(C+1):
                    for i in range(9):
                        for j in range(9):
                            if mincost>dp[X-1][c][i][j]:
                                mincost = dp[X-1][c][i][j]
                                mincij = (c, i, j)
                costs += mincost
                c, i, j = mincij
                new_plan = [j, i]
                for x in range(X-2, -1, -1):
                    found = False
                    for i_, j_ in [(a, b) for a in range(9) for b in range(9)]:
                        c_ = c-(i!=i_)-(j!=j_)
                        if c_>=0 and  dp[x][c_][i_][j_]+Costs[m][i][0]+Costs[m][j][1]==dp[x+1][c][i][j]:
                            c, i, j = c_, i_, j_
                            new_plan.append(j)
                            new_plan.append(i)
                            found = True
                            break
                    if not found:
                        raise ValueError
                new_plan = new_plan[::-1]
                new_works.append(new_plan)
            # print(1, new_plan)
            # print(dp)
            # raise ValueError
            # print(res.score_ignore_c, round(10**9*(10-math.log(costs/X, 10))))
            # print(1, new_works)
            score = cal_score(new_works)
            if score>best_score:
                best_score = score
                best_works = new_works
        state = State(best_works)
        # print(2, state.works)
        return state



def solve():
    results = Results()
    with open("./res.txt", "w") as f:
        f.write(f"{C}\n")
        for case in range(E):
            state = results.get_new_state(case)
            # anss = [[random.randint(1, 9) for _ in range(2*X)] for _ in range(M)]
            # anss = [[random.randint(7, 9)]*(2*X) for _ in range(M)]
            # 出力、結果保存
            for work in state.works:
                print("".join(map(lambda x: str(x+1), work)))
            score, v, d = map(int, input().split())
            RDRO = [[list(map(float, input().split())) for _ in range(X)] for _ in range(M)]
            result = Result(state, score, v, d, RDRO)
            results.add_result(result)
            f.write(f"{score} {v} {d} {result.score_ignore_c}\n")

solve()
