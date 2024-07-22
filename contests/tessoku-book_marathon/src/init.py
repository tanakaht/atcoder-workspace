import math
import random
from time import time

TS = time()
TL = 1.8

# [a, b)
def randint(a, b):
    return int(random.random()*(b-a))+a


# 連結前提
class State:
    def __init__(self, *args):
        pass

    @classmethod
    def get_initial_state(cls, **kwargs):
        raise NotImplementedError

    def move(self, *args):
        raise NotImplementedError

    def unmove(self, *args):
        raise NotImplementedError

    def get_neighbor(self, *args):
        raise NotImplementedError

    # TODO: 実装次第で修正
    def get_neighbor_with_score(self, *args):
        params = self.get_neighbor()
        self.move(params)
        score = self.get_score()
        self.unmove(params)
        return params, score

    def get_score(self, **params):
        raise NotImplementedError

    def print(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


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
            params, new_score = state.get_neighbor()
            temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL
            if math.exp((new_score-score)/temp) > random.random():
                state.move(params)
                score = new_score
                self.history.append((cnt, score, state.copy()))
        return state

    def beam_search(self, TL=1.8, beam_width=10):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

init_state = State.get_initial_state()
opt = Optimizer()
# best_state = opt.climbing()
best_state = opt.simanneal()
best_state.print()
# opt.visualize()
