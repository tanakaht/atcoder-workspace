import math
import random
from time import time
import sys
import heapq
input = sys.stdin.readline

logfile = open("./log.txt", "w")
def logprint(*args):
    print(*args, file=logfile)
TS = time()
TL = 4.5
# [a, b)
def randint(a, b):
    return int(random.random()*(b-a))+a

class Worker:
    def __init__(self, worker_id, v_init, Lmax, *job_types) -> None:
        self.worker_id = worker_id
        self.v_init = v_init
        self.Lmax = Lmax
        self.job_types = set(job_types)
        self.v = v_init
        # self.jobs = []
        # 移動完了までの時間
        self.move_cnt = 0
        self.acts = [None]*1000

    def clear(self):
        self.v = self.v_init
        self.acts = [None]*1000

class Job:
    def __init__(self, job_id, job_type, Ntask, v, rewards, depends) -> None:
        self.job_id = job_id
        self.job_type = job_type
        self.Ntask = Ntask
        self.rest_task = Ntask
        self.v = v
        self.depends = depends
        self.depend_cnt = len(depends)
        self.rewards = [0]*1000
        pre_t, pre_y = rewards[0], rewards[1]
        self.end_turn = math.inf
        it = iter(rewards[2:])
        for t, y in zip(it, it):
            t -= 1
            for i in range(pre_t, t):
                self.rewards[i] = (y-pre_y)*(i-pre_t)/(t-pre_t)+pre_y
            pre_t, pre_y = t, y
        self.max_rewards = max(self.rewards)

    def clear(self):
        self.depend_cnt = len(self.depends)
        self.end_turn = math.inf
        self.rest_task = self.Ntask


Tmax = int(input())
Nv, Ne = map(int, input().split())
UVD = [list(map(int, input().split())) for _ in range(Ne)]
Nworker = int(input())
workers = []
for worker_id in range(Nworker):
    v_init, Lmax, Njobtypes, *job_types = map(int, input().split())
    v_init -= 1
    worker = Worker(worker_id, v_init, Lmax, *job_types)
    workers.append(worker)
Njob = int(input())
jobs = []
for job_id in range(Njob):
    _, job_type, Ntask, vjob = map(int, input().split())
    vjob -= 1
    rewards = list(map(int, input().split()))[1:]
    depends = list(map(lambda x: int(x)-1, input().split()))[1:]
    job = Job(job_id, job_type, Ntask, vjob, rewards, depends)
    jobs.append(job)

G = [[] for _ in range(Nv)]
for u, v, d in UVD:
    u -= 1
    v -= 1
    G[u].append((v, d))
    G[v].append((u, d))

dists = [[math.inf]*Nv for _ in range(Nv)]
for start_node in range(Nv):
    q = [(0, start_node)]
    dists_ = dists[start_node]
    dists_[start_node] = 0
    appeared = [False]*Nv
    while q:
        d, u = heapq.heappop(q)
        if appeared[u]:
            continue
        appeared[u] = True
        for v, c in G[u]:
            d_ = d+c
            if dists_[v] > d_:
                dists_[v] = d_
                heapq.heappush(q, (d_, v))


class State_phase:
    def __init__(self, workers, jobs, dists, target_job_ids, duration):
        self.workers = workers
        self.jobs = jobs
        self.dists = dists
        self.target_job_ids = target_job_ids
        self.duration = duration
        self.job_allocations = [[] for _ in range(len(self.workers))]
        pass

    def initial_ans(self):
        pass

    def simanneal(self, initial_state, TL_=TL, save_history=False):
        TS = time()
        state = initial_state.copy()
        score = state.get_score()
        start_temp = 100000
        end_temp = 0.1
        cnt = 0
        if save_history:
            self.history.append((cnt, score, state))
        while True:
            cur_time =time()
            if cur_time-TS>=TL_:
                break
            cnt += 1
            params, new_score = state.get_neighbor_with_score()
            temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL
            if (new_score>=score or math.exp((new_score-score)/temp) > random.random()):
                state.move(*params)
                score = new_score
                self.history.append((cnt, score, state.copy()))
        return state


class State:
    def __init__(self, workers, jobs, dists):
        self.workers = workers
        self.jobs = jobs
        self.dists = dists
        self.job_allocations = [[] for _ in range(len(self.workers))]

    @classmethod
    def get_initial_state(cls, **kwargs):
        ret = State(workers, jobs, dists)
        # 貪欲割り当て, それぞれ適当なscore最大化するjobを一つずつ割り当てる
        rest_jobs = set(range(len(jobs)))
        for turn in range(Tmax):
            for worker_id in range(Nworker):
                worker = ret.workers[worker_id]
                if worker.acts[turn] is not None:
                    continue
                best_job_id, best_score = None, -math.inf
                for job_id in rest_jobs:
                    job = jobs[job_id]
                    # 実行可能
                    if job.job_type not in worker.job_types:
                        continue
                    # 依存解決ずみ
                    is_ok = True
                    for depend_job_id in job.depends:
                        if not (jobs[depend_job_id].end_turn<=turn):
                            is_ok = False
                            break
                    if not is_ok:
                        continue
                    # スコア計算
                    move_turn = dists[worker.v][job.v]
                    exe_turn = math.ceil(job.rest_task/worker.Lmax)
                    start_turn = min(Tmax, turn+move_turn)
                    end_turn = min(turn+move_turn+exe_turn, Tmax)
                    n_task = min(job.rest_task, (end_turn-start_turn)*worker.Lmax)
                    if n_task<=0:
                        continue
                    rewards = (job.rewards[start_turn], job.rewards[end_turn-1])
                    if min(rewards)==0:
                        continue
                    # 移動ターンが少ない
                    # 収穫時期が良い
                    # rewardsが多い
                    score = (rewards[0]+rewards[1])/2*n_task/(move_turn+exe_turn) # /((move_turn+1)**4)
                    # 更新
                    if score > best_score:
                        best_score = score
                        best_job_id = job_id
                if best_job_id is None:
                    continue
                job_id = best_job_id
                job = ret.jobs[job_id]
                cur_worker_v = worker.v
                # 移動
                for turn_ in range(turn, turn+dists[cur_worker_v][job.v]):
                    worker.acts[turn_] = ("move", job.v)
                worker.v = job.v
                # job実行
                end_turn = min(Tmax, turn+dists[cur_worker_v][job.v]+math.ceil(job.rest_task/worker.Lmax))
                for turn_ in range(turn+dists[cur_worker_v][job.v], end_turn):
                    n = min(worker.Lmax, job.rest_task)
                    worker.acts[turn_] = ("execute", job_id, n)
                    job.rest_task -= n
                job.end_turn = end_turn
                rest_jobs.remove(job_id)
                ret.job_allocations[worker_id].append(job_id)
        # やりきれなかったものも適当に割り当てる
        rest_job_allocation_cnt = [0]*len(workers)
        worker_id = 0
        for job_id in list(rest_jobs):
            best_worker_id, best_cnt = None, math.inf
            for worker_id, worker in enumerate(workers):
                if jobs[job_id].job_type in worker.job_types:
                    if rest_job_allocation_cnt[worker_id]<best_cnt:
                        best_worker_id = worker_id
                        best_cnt = rest_job_allocation_cnt[worker_id]
            if best_worker_id is None:
                continue
            ret.job_allocations[best_worker_id].append(job_id)
            rest_job_allocation_cnt[best_worker_id] += 1
        return ret

    def move(self, params):
        # jobのswap ("swap", i, j, x, y)
        # worker:iのx番目のjobとworker: jのy番目を入れ替え
        if params[0]=="swap":
            i, j, x, y = params[1:]
            job_id_x = self.job_allocations[i][x]
            job_id_y = self.job_allocations[j][y]
            self.job_allocations[i][x], self.job_allocations[j][y] = job_id_y, job_id_x
        # jobの移動 ("move", i, j, x, y)
        # worker:iのx番目のjobをworker: jのy番目に挿入
        elif params[0]=="move":
            i, j, x, y = params[1:]
            job_id = self.job_allocations[i].pop(x)
            self.job_allocations[j].insert(y, job_id)
        # 休憩挿入 ("ins_rest", i, x)
        elif params[0]=="ins_rest":
            i, x = params[1:]
            self.job_allocations[i].insert(x, -1)
        # 休憩pop ("pop_rest", i, x)
        elif params[0]=="pop_rest":
            i, x = params[1:]
            job_id = self.job_allocations[i].pop(x)
            assert job_id==-1
        else:
            pass

    def unmove(self, params):
        if params[0]=="swap":
            self.move(params)
        # jobの移動 ("move", i, j, x, y)
        # worker:iのx番目のjobをworker: jのy番目に挿入
        elif params[0]=="move":
            i, j, x, y = params[1:]
            self.move(("move", j, i, y, x))
        # 休憩挿入 ("ins_rest", i, x)
        elif params[0]=="ins_rest":
            i, x = params[1:]
            self.move(("pop_rest", i, x))
        # 休憩pop ("pop_rest", i, x)
        elif params[0]=="pop_rest":
            i, x = params[1:]
            self.move(("ins_rest", i, x))
        else:
            pass

    # 期間swap
    # 2opt
    # 期間移動
    def get_neighbor(self, *args):
        flg = randint(0, 100)
        # swap
        if flg<=70:
            for _ in range(100):
                i, j = randint(0, len(self.workers)), randint(0, len(self.workers))
                if min(len(self.job_allocations[i]), len(self.job_allocations[j]))==0:
                    continue
                x, y = randint(0, len(self.job_allocations[i])), randint(0, len(self.job_allocations[j]))
                if i==j and x==y:
                    continue
                # xとyが入れ替え可能かチェック
                jobid_x, jobid_y = self.job_allocations[i][x], self.job_allocations[j][y]
                if jobid_x==-1 and jobid_y==-1:
                    continue
                if jobid_x!=-1 and self.jobs[jobid_x].job_type not in self.workers[j].job_types:
                    continue
                if jobid_y!=-1 and self.jobs[jobid_y].job_type not in self.workers[i].job_types:
                    continue
                return ("swap", i, j, x, y)
            return ("none", )
        # move
        elif flg<=90:
            for _ in range(100):
                i, j = randint(0, len(self.workers)), randint(0, len(self.workers))
                x, y = randint(0, len(self.job_allocations[i])), randint(0, len(self.job_allocations[j])+(i!=j))
                if len(self.job_allocations[i])==0:
                    continue
                if i==j and x==y:
                    continue
                # xとyが入れ替え可能かチェック
                jobid_x = self.job_allocations[i][x]
                if jobid_x!=-1 and self.jobs[jobid_x].job_type in self.workers[j].job_types:
                    continue
                return ("move", i, j, x, y)
            return ("none", )
        # ins_rest
        elif flg<=92:
            i = randint(0, len(self.workers))
            x = randint(0, len(self.job_allocations[i])+1)
            return ("ins_rest", i, x)
        # pop_rest
        elif flg<=100:
            total_rest = sum([sum([x==-1 for x in job_allocation]) for job_allocation in self.job_allocations])
            if total_rest==0:
                return ("none", )
            target_rest = randint(0, total_rest)
            rest_cnt = 0
            for i in range(len(self.workers)):
                for x, job_id in enumerate(self.job_allocations[i]):
                    if job_id==-1:
                        if target_rest==rest_cnt:
                            return ("pop_rest", i, x)
                        rest_cnt += 1

    # TODO: 実装次第で修正
    def get_neighbor_with_score(self, *args):
        params = self.get_neighbor()
        self.move(params)
        score = self.get_score()
        self.unmove(params)
        return params, score

    def get_score(self, **params):
        for worker in self.workers:
            worker.clear()
        for job in self.jobs:
            job.clear()
        job_idx = [0]*len(self.workers)
        score = 0
        for turn in range(Tmax):
            for worker_id in range(Nworker):
                worker = self.workers[worker_id]
                if (worker.acts[turn] is not None) or (len(self.job_allocations[worker_id])<=job_idx[worker_id]):
                    continue
                job_id = self.job_allocations[worker_id][job_idx[worker_id]]
                # -1は休憩
                if job_id == -1:
                    job_idx[worker_id] += 1
                    continue
                job = self.jobs[job_id]
                # 実行可能check
                if job.job_type not in worker.job_types:
                    return 0
                # job依存解決まで待機
                is_ok = True
                for depend_job_id in job.depends:
                    if not (jobs[depend_job_id].end_turn<=turn):
                        is_ok = False
                        break
                if not is_ok:
                    continue
                cur_worker_v = worker.v
                move_turn = dists[worker.v][job.v]
                exe_turn = math.ceil(job.rest_task/worker.Lmax)
                start_turn = min(Tmax, turn+move_turn)
                end_turn = min(turn+move_turn+exe_turn, Tmax)
                # 移動
                for turn_ in range(turn, start_turn):
                    worker.acts[turn_] = ("move", job.v)
                worker.v = job.v
                # 実行可能チェック
                rewards = (job.rewards[min(start_turn, Tmax-1)], job.rewards[end_turn-1])
                if job.rest_task<=0 or min(rewards)==0:
                    # 不可なら休憩
                    continue
                # job実行
                for turn_ in range(start_turn, end_turn):
                    n = min(worker.Lmax, job.rest_task)
                    assert n>0
                    worker.acts[turn_] = ("execute", job_id, n)
                    job.rest_task -= n
                    score += job.rewards[turn_]*n
                job.end_turn = end_turn
                job_idx[worker_id] += 1
        return score

    def print(self):
        # self.get_score()
        for i in range(Tmax):
            for worker in self.workers:
                act = worker.acts[i]
                if act is None:
                    print("stay")
                else:
                    if act[0]=="stay":
                        print(*act)
                    elif act[0]=="move":
                        print(act[0], act[1]+1)
                    elif act[0] == "execute":
                        print(act[0], act[1]+1, act[2])
                    else:
                        raise ValueError


    def copy(self):
        ret = State(workers, jobs, dists)
        ret.job_allocations = [list(x) for x in self.job_allocations]
        return ret


class Optimizer:
    def __init__(self, verbose=False):
        self.history = []

    def climbing(self, initial_state, TL_=TL, save_history=False):
        state = initial_state.copy()
        score = state.get_score()
        cnt = 0
        if save_history:
            self.history.append((cnt, score, state))
        TS_ = time()
        while time()-TS_<TL_:
            cnt += 1
            params, new_score = state.get_neighbor_with_score()
            if new_score >= score:
                state.move(params)
                score = new_score
                if save_history:
                    self.history.append((cnt, score, state.copy()))
        return state

    def simanneal(self, initial_state, TL_=TL, save_history=False):
        TS = time()
        state = initial_state.copy()
        score = state.get_score()
        best_state, best_score = state.copy(), score
        start_temp = 100000
        end_temp = 0.1
        cnt = 0
        if save_history:
            self.history.append((cnt, score, state))
        while True:
            cur_time =time()
            if cur_time-TS>=TL_:
                break
            cnt += 1
            params, new_score = state.get_neighbor_with_score()
            if params[0]=="none":
                continue
            temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL
            if (new_score>=score or math.exp((new_score-score)/temp) > random.random()):
                # if score!=new_score:
                #     print(new_score, score)
                state.move(params)
                score = new_score
                self.history.append((cnt, score, state.copy()))
            if score > best_score:
                best_state, best_score = state.copy(), score
        return best_state

    def beam_search(self, TL=1.8, beam_width=10):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

init_state = State.get_initial_state()
# init_state.get_score()
# init_state.print()
opt = Optimizer()
# best_state = opt.climbing(init_state)
best_state = opt.simanneal(init_state)
best_state.get_score()
best_state.print()
# print(init_state.get_score(), best_state.get_score())
# print(best_state.get_score(), init_state.get_score())
# opt.visualize()
