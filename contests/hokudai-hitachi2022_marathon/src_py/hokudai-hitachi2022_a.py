import math
import random
from time import time
import sys
import heapq
from typing import List
input = sys.stdin.readline

logfile = open("./log.txt", "w")
def logprint(*args):
    print(*args, file=logfile)

TS = time()
TL = 4.5
N_PHASE = 4
PHASE_BORDER_TURN = 10
# [a, b)
def randint(a, b):
    return int(random.random()*(b-a))+a

def phase2duration(phase, Tmax):
    if phase==0:
        return [0, 50]
    elif phase==N_PHASE-1:
        return [Tmax-50, Tmax]
    else:
        return [50+(Tmax-100)*(phase-1)//(N_PHASE-2)-PHASE_BORDER_TURN, 50+(Tmax-100)*phase//(N_PHASE-2)+PHASE_BORDER_TURN]

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
        self.last_move_turn = 0

    def clear(self):
        self.v = self.v_init
        self.acts = [None]*1000
        self.last_move_turn = 0

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
        self.phasewise_rewards_ = [self.Ntask*min(self.rewards[phase2duration(phase, Tmax)[0]:phase2duration(phase, Tmax)[1]]) for phase in range(N_PHASE)]  # 戻すよう
        self.phasewise_rewards = [self.Ntask*min(self.rewards[phase2duration(phase, Tmax)[0]:phase2duration(phase, Tmax)[1]]) for phase in range(N_PHASE)]  # いじるよう

    def clear(self):
        self.depend_cnt = len(self.depends)
        self.end_turn = math.inf
        self.rest_task = self.Ntask


Tmax = int(input())
Nv, Ne = map(int, input().split())
UVD = [list(map(int, input().split())) for _ in range(Ne)]
Nworker = int(input())
workers: List[Worker] = []
for worker_id in range(Nworker):
    v_init, Lmax, Njobtypes, *job_types = map(int, input().split())
    v_init -= 1
    worker = Worker(worker_id, v_init, Lmax, *job_types)
    workers.append(worker)
Njob = int(input())
jobs: List[Job] = []
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

target_v = set([job.v for job in jobs]+[worker.v for worker in workers])
dists = [[math.inf]*Nv for _ in range(Nv)]
for start_node in range(Nv):
    if start_node not in target_v:
        continue
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


"""
- 依存解決されていないものは無視
- Scoreはphase中の最小score (平均2倍あるが一旦無視)
    - 新しく解禁されるjobのscoreも重みつけて加算
    - Minがmaxとの差がでかいときは入れない
- dist+実行時間(workerごと)を使って、score最大なworkerへの割り当てを決める。(phase+Xターンくらい?)
    - 適当に巡回セールスマンっぽく焼けそう
    - 期間内のscoreも入れれたら入れる?
    - 状態
        - 割り当て(順序込み)
        - restもあり
    - 近傍
        - worker間
            - jobのswap(区間)
        - worker内
            - jobのswap(区間)
    - score
        - 終了前のjobのスコアのsum+残っているターン(軽く)
    - 初期解
        - スコア大きい順に終わらせるのが一番早い人にアサインする
            - 移動距離が長すぎる場合はアサインしない
"""
class PhaseJobAllocator:
    def __init__(self, workers, jobs: List[Job], ent_dists, target_job_ids, phase):
        self.workers = workers
        self.jobs = jobs
        self.ent_dists = ent_dists
        self.target_job_ids = target_job_ids
        self.phase = phase
        self.duration = phase2duration(phase, Tmax)
        self.job_allocations = [[] for _ in range(len(self.workers))]
        self.score = [jobs[job_id].phasewise_rewards[phase] for job_id in self.target_job_ids]
        self.dists = [[[math.inf]*(len(target_job_ids)+1) for _ in range(len(target_job_ids)+1)] for _ in range(Nworker)]
        for worker_id in range(Nworker):
            worker: Worker = self.workers[worker_id]
            for i, job_id1 in enumerate(self.target_job_ids):
                job1: Job = self.jobs[job_id1]
                if job1.job_type not in worker.job_types:
                    continue
                self.dists[worker_id][len(target_job_ids)][i] = self.ent_dists[worker.v][job1.v] + max(0, worker.last_move_turn-self.duration[0])
                for j, job_id2 in enumerate(self.target_job_ids):
                    job2: Job = self.jobs[job_id2]
                    if job2.job_type not in worker.job_types:
                        continue
                    self.dists[worker_id][i][j] = self.ent_dists[job1.v][job2.v]
        self.task_turn = [[math.ceil(self.jobs[job_id].rest_task/worker.Lmax) if self.jobs[job_id].job_type in worker.job_types else math.inf for job_id in self.target_job_ids] for worker in self.workers]
        self.internal_job_score = [self.jobs[job_id].phasewise_rewards[self.phase] for internal_job_id, job_id in enumerate(self.target_job_ids)]
        self.initial_allocation()

    def get_worker_last_move_turn(self, worker_id, job_allocation=None):
        job_allocation = job_allocation if job_allocation is not None else self.job_allocations[worker_id]
        worker = self.workers[worker_id]
        last_move_turn = max(self.duration[0], worker.last_move_turn)
        worker_v = worker.v
        for internal_job_id in job_allocation:
            job_id = self.target_job_ids[internal_job_id]
            job = self.jobs[job_id]
            last_move_turn += self.ent_dists[worker_v][job.v] + self.task_turn[worker_id][internal_job_id]
            worker_v = job.v
        return last_move_turn

    def initial_allocation(self):
        worker_job_list = [(worker_id, internal_job_id) for worker_id in range(len(self.workers)) for internal_job_id in range(len(self.target_job_ids)) if self.task_turn[worker_id][internal_job_id]<=duration[1]-duration[0]]
        is_allocated = [False]*len(self.target_job_ids)
        worker_v = [-1 for worker in self.workers]
        for worker_id, internal_job_id in sorted(worker_job_list, key=lambda x: -(self.internal_job_score[x[1]])/(self.dists[x[0]][-1][x[1]]+self.task_turn[x[0]][x[1]])):
            if is_allocated[internal_job_id]:
                continue
            end_turn = self.get_worker_last_move_turn(worker_id, job_allocation=self.job_allocations[worker_id]+[internal_job_id])
            if end_turn >= self.duration[1]:
                continue
            self.job_allocations[worker_id].append(internal_job_id)
            is_allocated[internal_job_id] = True
            worker_v[worker_id] = internal_job_id
        return
        for internal_job_id, job_id in sorted(enumerate(self.target_job_ids), key=lambda x: -self.internal_job_score[x[0]]):
            job = self.jobs[job_id]
            best_worker_id = None
            best_worker_metrics = (math.inf, math.inf) # 移動ターン, 終了ターン
            for worker_id, worker in enumerate(self.workers):
                if job.job_type not in worker.job_types:
                    continue
                job_allocation = self.job_allocations[worker_id]
                worker_v = self.jobs[self.target_job_ids[job_allocation[-1]]].v if len(job_allocation)!=0 else worker.v
                move_turn = self.ent_dists[worker_v][job.v]
                end_turn = self.get_worker_last_move_turn(worker_id, job_allocation=self.job_allocations[worker_id]+[internal_job_id])
                if end_turn >= self.duration[1]:
                    continue
                if move_turn >= 50:
                    continue
                # 移動し過ぎなら強制入れ替え
                if best_worker_metrics[0]>=15 and move_turn<15:
                    best_worker_id = worker_id
                    best_worker_metrics = (move_turn, end_turn)
                # そうでなければend_turnで比較
                elif end_turn+(move_turn/Tmax)<best_worker_metrics[1]+(best_worker_metrics[0]/Tmax):
                    best_worker_id = worker_id
                    best_worker_metrics = (move_turn, end_turn)
            if best_worker_id is not None:
                self.job_allocations[best_worker_id].append(internal_job_id)

    def do_work(self, worker, job):
        worker.last_move_turn = max(self.duration[0], worker.last_move_turn)
        worker.last_move_turn += self.ent_dists[worker.v][job.v] + math.ceil(job.rest_task/worker.Lmax)
        worker.v = job.v
        job.end_turn = worker.last_move_turn

    def get_and_assign_job_allocation(self, TL):
        self.simanneal(TL_=TL)
        # do_work
        for worker_id, job_allocation in enumerate(self.job_allocations):
            for internal_job_id in job_allocation:
                job = self.jobs[self.target_job_ids[internal_job_id]]
                worker = self.workers[worker_id]
                self.do_work(worker, job)
        ret = [[self.target_job_ids[internal_job_id] for internal_job_id in job_allocation] for job_allocation in self.job_allocations]
        return ret

    def get_score(self):
        for worker_id, job_allocation in enumerate(self.job_allocations):
            pass
        return 1

    def move(self, *params):
        params = self.get_neighbor()
        self.move(params)
        score = self.get_score()
        self.unmove(params)
        return params, score

    # jobのswap
    # 2opt
    def get_neighbor(self):
        pass

    def get_neighbor_with_score_diff(self):
        params = self.get_neighbor()
        return (1, 1)

    def simanneal(self, TL_=TL, save_history=False):
        TS = time()
        best_allocation = [[x for x in allocation] for allocation in self.job_allocations]
        # 初期スコア計算
        last_move_turns = [self.get_worker_last_move_turn(worker_id) for worker_id in range(len(self.workers))]
        score = sum([self.internal_job_score[internal_job_id] for job_allocation in self.job_allocations for internal_job_id in job_allocation])
        best_score = score
        rest_jobs = list(range(len(self.target_job_ids)))
        for job_allocation in self.job_allocations:
            for internal_job_id in job_allocation:
                rest_jobs.remove(internal_job_id)
        start_temp = 100000
        end_temp = 0.1
        while True:
            cur_time =time()
            if cur_time-TS>=TL_:
                break
            flg = randint(0, 12)
            # jobswap
            if flg<=4:
                i, j = max(-1, randint(-len(self.workers), len(self.workers))), max(-1, randint(-len(self.workers), len(self.workers)))
                i, j = min(i, j), max(i, j)
                if i==-1 and j==-1:
                    continue
                job_allocation_i = self.job_allocations[i] if i!=-1 else rest_jobs
                job_allocation_j = self.job_allocations[j] if j!=-1 else rest_jobs
                if min(len(job_allocation_i), len(job_allocation_j))==0:
                    continue
                x, y = randint(0, len(job_allocation_i)), randint(0, len(job_allocation_j))
                internal_job_id_x, internal_job_id_y = job_allocation_i[x], job_allocation_j[y]
                if i==j:
                    continue
                else:
                    # ターン内に終わるかチェック
                    if i!=-1:
                        end_turn_i = self.get_worker_last_move_turn(i, job_allocation_i[:x]+[internal_job_id_y]+job_allocation_i[x+1:])
                        if end_turn_i>=self.duration[1]:
                            continue
                    end_turn_j = self.get_worker_last_move_turn(j, job_allocation_j[:y]+[internal_job_id_x]+job_allocation_j[y+1:])
                    if end_turn_j>=self.duration[1]:
                        continue
                    if i==-1:
                        pre_score = score
                        new_score = score+self.internal_job_score[internal_job_id_x]-self.internal_job_score[internal_job_id_y]
                    else:
                        pre_score = last_move_turns[i]+last_move_turns[j]
                        new_score = end_turn_i+end_turn_j
                    temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL
                    if (new_score>=pre_score or math.exp((new_score-pre_score)/temp) > random.random()):
                        # print("swap", i, j, x, y, new_score, pre_score)
                        # 遷移
                        job_allocation_i[x], job_allocation_j[y] = job_allocation_j[y], job_allocation_i[x]
                        # score更新
                        if i==-1:
                            score = new_score
                        else:
                            last_move_turns[i] = end_turn_i
                        last_move_turns[j] = end_turn_j
            # jobmove
            elif flg <= 8:
                i, j = max(-1, randint(-len(self.workers)*4, len(self.workers))), max(-1, randint(-len(self.workers), len(self.workers)))
                if i==-1 and j==-1:
                    continue
                job_allocation_i = self.job_allocations[i] if i!=-1 else rest_jobs
                job_allocation_j = self.job_allocations[j] if j!=-1 else rest_jobs
                if len(job_allocation_i)==0:
                    continue
                x, y = randint(0, len(job_allocation_i)), randint(0, len(job_allocation_j))
                internal_job_id_x = job_allocation_i[x]
                if i==j:
                    continue
                else:
                    # ターン内に終わるかチェック
                    if i!=-1:
                        end_turn_i = self.get_worker_last_move_turn(i, job_allocation_i[:x]+job_allocation_i[x+1:])
                        if end_turn_i>=self.duration[1]:
                            continue
                    if j!=-1:
                        end_turn_j = self.get_worker_last_move_turn(j, job_allocation_j[:y]+[internal_job_id_x]+job_allocation_j[y:])
                        if end_turn_j>=self.duration[1]:
                            continue
                    if i==-1:
                        pre_score = score
                        new_score = score+self.internal_job_score[internal_job_id_x]
                    elif j==-1:
                        pre_score = score
                        new_score = score-self.internal_job_score[internal_job_id_x]
                    else:
                        pre_score = last_move_turns[i]+last_move_turns[j]
                        new_score = end_turn_i+end_turn_j
                    temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL
                    if (new_score>=pre_score or math.exp((new_score-pre_score)/temp) > random.random()):
                        # print("move", i, j, x, y, new_score, pre_score)
                        # 遷移
                        job_allocation_j.insert(y, job_allocation_i.pop(x))
                        # score更新
                        if i==-1:
                            score = new_score
                            last_move_turns[j] = end_turn_j
                        elif j==-1:
                            score = new_score
                            last_move_turns[i] = end_turn_i
                        else:
                            last_move_turns[i] = end_turn_i
                            last_move_turns[j] = end_turn_j
            # 2-opt
            else:
                pass


class JobAllocation2Ans:
    def __init__(self, workers, jobs, dists, job_allocations):
        self.workers = workers
        self.jobs = jobs
        self.dists = dists
        self.job_allocations = job_allocations
        # self.job_allocations = [[] for _ in range(len(self.workers))]

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

    def allocate_restjob(self):
        # 貪欲割り当て, それぞれ適当なscore最大化するjobを一つずつ割り当てる
        rest_jobs = set(range(len(jobs)))
        for job_allocation in self.job_allocations:
            for job_id in job_allocation:
                rest_jobs.remove(job_id)
        job_idx = [0]*len(self.workers)
        for turn in range(Tmax):
            for worker_id in range(Nworker):
                worker = self.workers[worker_id]
                if worker.acts[turn] is not None:
                    continue
                if job_idx[worker_id]<len(self.job_allocations[worker_id]):
                    best_job_id = self.job_allocations[worker_id][job_idx[worker_id]]
                    if best_job_id is None:
                        continue
                    job_id = best_job_id
                    job = self.jobs[job_id]
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
                    job_idx[worker_id] += 1
                else:
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
                    job = self.jobs[job_id]
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
                    self.job_allocations[worker_id].append(job_id)
                    job_idx[worker_id] += 1
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
            self.job_allocations[best_worker_id].append(job_id)
            rest_job_allocation_cnt[best_worker_id] += 1
        return self

    def print(self):
        self.get_score()
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

job_allocations = [[] for _ in range(Nworker)]
for phase in range(N_PHASE):
    duration = phase2duration(phase, Tmax)
    target_job_ids = []
    for job_id, job in enumerate(jobs):
        flg = job.end_turn >= duration[1]
        for depend_job_id in job.depends:
            flg = flg and (jobs[depend_job_id].end_turn<duration[0])
        if flg and job.phasewise_rewards[phase]>0:
            target_job_ids.append(job_id)
    job_allocator = PhaseJobAllocator(workers, jobs, dists, target_job_ids, phase)
    for worker_id, job_allocation in enumerate(job_allocator.get_and_assign_job_allocation(1)):
        job_allocations[worker_id] += job_allocation
printer = JobAllocation2Ans(workers, jobs, dists, job_allocations)
# 残ったジョブを適当に割り当てる
printer.allocate_restjob()
printer.print()
# print(job_allocations)
