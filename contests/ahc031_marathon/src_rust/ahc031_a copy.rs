#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
use std::vec;
use itertools::Itertools;
use num::{range, ToPrimitive};
use num_integer::Roots;
use proconio::{input, marker::Chars};
use rand_core::block;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fmt::Display;
use std::cmp::{Reverse, Ordering};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;
use ndarray::prelude::*;
use num::BigUint;


#[allow(unused_variables)]
const W: usize = 1000;

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    D: usize,
    N: usize,
    A: Vec<Vec<usize>>,
    rest_rates: Vec<usize>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            _: usize,
            D: usize,
            N: usize,
            A: [[usize; N]; D], // a is Vec<i32>, n-array.
            //m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let mut rest_rates: Vec<usize> = vec![];
        for d in 0..D {
            rest_rates.push((W * W - A[d].iter().sum::<usize>()) / N);
        }
        Self { D, N, A, rest_rates }
    }
}

struct Neighbor{
    x: usize
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    rects: Vec<(usize, usize, usize, usize)>, // from_d, to_d, width, is_tate
    rest_rect_cnt: Vec<usize>,
    rest_rect_idx: Vec<Vec<usize>>,
    cur_w: Vec<usize>,
    cur_h: Vec<usize>,
    rest_needed_space: Vec<usize>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut rects = vec![];
        let mut rest_rect_cnt = vec![input.N; input.D];
        let mut rest_rect_idx: Vec<Vec<usize>> = vec![(0..input.N).collect(); input.D];
        let mut rest_needed_space: Vec<usize> = vec![];
        for a in input.A.iter(){
            rest_needed_space.push(a.iter().sum());
        }
        // for n in 0..input.N{
        //     rects.push((0, input.D, 2*W/(input.N+2), n%2));
        //     rest_rect_cnt -= input.N;
        // }
        let mut ret = Self {rects, rest_rect_cnt, rest_rect_idx, cur_w: vec![0; input.D], cur_h: vec![0; input.D], rest_needed_space};
        ret.greedy_solve(input, 0, input.D);
        ret
    }

    fn greedy_solve(&mut self, input: &Input, from_t: usize, to_t: usize){
        if from_t==to_t || self.rest_rect_cnt[from_t]==0{
            return;
        }
        if self.rest_rect_cnt[from_t]==1{
            self.put_rect(input, from_t, to_t, W-self.cur_w[from_t], 0);
        }
        // できるだけ長い時間帯でトライ
        for t_ in 0..to_t-from_t{
            let t = to_t-t_;
            // 一番良い幅を探す
            let mut best_param = (0, 0);
            let mut best_score = usize::MAX;
            // 横おき
            for w in 1..W-self.cur_w[from_t]{
                let area = w*(W-self.cur_h[from_t]);
                let mut score = W-w;
                let mut is_valid = true;
                for d in from_t..t{
                    let mut daily_score = usize::MAX;
                    let mut daily_valid = false;
                    for i in self.rest_rect_idx[d].iter(){
                        let a = input.A[d][*i];
                        let rest_space = (W-self.cur_h[d])*(W-self.cur_w[d])-area;
                        // 残りがおけることが確定している or 充足率がほどほど
                        if a<=area && rest_space>=self.rest_needed_space[d]-a && (self.rest_needed_space[d]-a<=rest_space-std::cmp::min(W-self.cur_w[d]-w, W-self.cur_h[d])*self.rest_rect_cnt[d] || input.rest_rates[d]*8/10<=((rest_space-(self.rest_needed_space[d]-a))/(self.rest_rect_cnt[d]-1)) ){
                            daily_valid = true;
                            daily_score = std::cmp::min(daily_score, area-a);
                        }
                    }
                    if !daily_valid{
                        is_valid = false;
                        break;
                    }
                    score += daily_score;
                }
                if !is_valid{
                    continue;
                }
                if score<best_score{
                    best_score = score;
                    best_param = (w, 0);
                }
            }
            // 縦置き
            for w in 1..W-self.cur_h[from_t]{
                let area = w*(W-self.cur_w[from_t]);
                let mut score = W-w;
                let mut is_valid = true;
                for d in from_t..t{
                    let mut daily_score = usize::MAX;
                    let mut daily_valid = false;
                    for i in self.rest_rect_idx[d].iter(){
                        let a = input.A[d][*i];
                        let rest_space = (W-self.cur_h[d])*(W-self.cur_w[d])-area;
                        if a<=area && rest_space>=self.rest_needed_space[d]-a && (self.rest_needed_space[d]-a<=rest_space-std::cmp::min(W-self.cur_w[d], W-self.cur_h[d]-w)*self.rest_rect_cnt[d] || input.rest_rates[d]*8/10<=((rest_space-(self.rest_needed_space[d]-a))/(self.rest_rect_cnt[d]-1)) ){
                            daily_valid = true;
                            daily_score = std::cmp::min(daily_score, area-a);
                        }
                    }
                    if !daily_valid{
                        is_valid = false;
                        break;
                    }
                    score += daily_score;
                }
                if !is_valid{
                    continue;
                }
                if score<best_score{
                    best_score = score;
                    best_param = (w, 1);
                }
            }

            // 見つかったら置いて再帰
            if best_score!=usize::MAX{
                self.put_rect(input, from_t, t, best_param.0, best_param.1);
                self.greedy_solve(input, from_t, t);
                if t!=to_t{
                    self.greedy_solve(input, t, to_t);
                }
                return;
            }
        }
        // 何も見つからなかったら強制的に1日分おく
        for _ in 0..self.rest_rect_cnt[from_t]{
            self.put_rect(input, from_t, from_t+1, (W-self.cur_w[from_t])/self.rest_rect_cnt[from_t], 0);
        }
        self.greedy_solve(input, from_t+1, to_t);
    }


    // rest_rect_cnt: Vec<usize>,
    // rest_rect_idx: Vec<Vec<usize>>,
    // cur_w: Vec<usize>,
    // cur_h: Vec<usize>,
    // rest_needed_space: Vec<usize>,

    fn put_rect(&mut self, input: &Input, from_d: usize, to_d: usize, w: usize, is_tate: usize){
        self.rects.push((from_d, to_d, w, is_tate));
        for d in from_d..to_d{
            let area;
            if is_tate==0{
                self.cur_w[d] += w;
                area = w*(W-self.cur_h[d]);
            } else {
                self.cur_h[d] += w;
                area = w*(W-self.cur_w[d]);
            }
            let mut best_score = usize::MAX;
            let mut best_idx = 0;
            for i in self.rest_rect_idx[d].iter(){
                let a = input.A[d][*i];
                if a<=area{
                    let score = area-a;
                    if score<best_score{
                        best_score = score;
                        best_idx = *i;
                    }
                } else {
                    let score = (a-area)*1000;
                    if score<best_score{
                        best_score = score;
                        best_idx = *i;
                    }
                }
            }
            self.rest_rect_cnt[d] -= 1;
            self.rest_rect_idx[d].retain(|x| *x!=best_idx);
            self.rest_needed_space[d] -= input.A[d][best_idx];
        }
    }

    fn update(&mut self, params: &Neighbor){
    }

    fn undo(&mut self, params: &Neighbor){
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let n = input.N;
        let mut rng = rand::thread_rng();
        // let mode_flg = rng.gen::<usize>()%100;
        Neighbor{x: rng.gen::<usize>()%100}
    }

    fn get_score(&self, input: &Input) -> f64{
        1.0
    }

    fn print(&mut self, input: &Input){
        let mut widthes = vec![vec![]; input.D];
        for (from_d, to_d, w, is_tate) in self.rects.iter(){
            for i in *from_d..*to_d{
                widthes[i].push((*w, *is_tate));
            }
        }
        for d in 0..input.D{
            let mut cur_w = 0;
            let mut cur_h = 0;
            let mut ans = vec![None; input.N];
            let mut rects = vec![];
            for (w, is_tate) in widthes[d].iter(){
                if *is_tate==0{
                    let area = *w*(W-cur_h);
                    rects.push((cur_w, cur_h, cur_w+*w, W, area));
                    cur_w += w;
                } else {
                    let area = *w*(W-cur_w);
                    rects.push((cur_w, cur_h, W, cur_h+*w, area));
                    cur_h += w;
                }
            }
            rects.sort_by_key(|x| x.4);
            rects.reverse();
            for (i, a) in input.A[d].iter().enumerate().sorted_by_key(|x| x.1){
                let (x1, y1, x2, y2, area) = rects.pop().unwrap();
                ans[i] = Some((x1, y1, x2, y2));
            }
            for (x1, y1, x2, y2) in ans.iter().flatten(){
                println!("{} {} {} {}", x1, y1, x2, y2);
            }
        }
    }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.get_score(input);
    let mut cur_score = best_score;
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 10.0;
    let end_temp: f64 = 0.1;
    let mut temp = start_temp;
    let mut last_updated = 0;
    eprintln!("simanneal start at {}", timer.elapsed().as_secs_f64());
    loop {
        let elasped_time = timer.elapsed().as_secs_f64();
        if elasped_time >= limit{
            break;
        }
        all_iter += 1;
        let neighbor = state.get_neighbor(input);
        state.update(&neighbor);
        let new_score = state.get_score(input);
        let score_diff = new_score-cur_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            accepted_cnt += 1;
            //eprintln!("{} {} {:?}", cur_score, new_score, turns);
            cur_score = new_score;
            last_updated = all_iter;
            //state.print(input);
            if new_score>best_score{
                best_state = state.clone();
                best_score = new_score;
            }
        } else {
            state.undo(&neighbor);
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("score  : {}", best_state.get_score(input));
    eprintln!("");
    best_state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.8);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut init_state = State::init_state(input);
    let mut best_state = simanneal(input, init_state, timer, 0.0);
    best_state.print(input);
}
