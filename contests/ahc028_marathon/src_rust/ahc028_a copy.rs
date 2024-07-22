#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use std::process::exit;
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

#[derive(Debug, Clone, Copy)]
struct Pos{
    row: usize,
    col: usize,
}

impl Pos{
    fn new(row: usize, col: usize) -> Self{
        Self{row, col}
    }

    fn dist(self, other: Self) -> usize{
        let dx = (self.row as i64-other.row as i64).abs();
        let dy = (self.col as i64-other.col as i64).abs();
        (dx+dy) as usize
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    m: usize,
    si: usize,
    sj: usize,
    pos2char: Vec<Vec<char>>,
    char2pos: HashMap<char, Vec<Pos>>,
    t: Vec<Vec<char>>,
    start_idxs: Vec<Vec<usize>>,
    dists: Vec<Vec<Vec<(usize, usize)>>>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            m: usize,
            si: usize,
            sj: usize,
            A: [Chars; n],
            t: [Chars; m],
            //a: [i32; n], // a is Vec<i32>, n-array.
            //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let mut char2pos = HashMap::new();
        for (i, x) in A.iter().enumerate(){
            for (j, &c) in x.iter().enumerate(){
                char2pos.entry(c).or_insert(vec![]).push(Pos::new(i, j));
            }
        }
        let mut start_idxs = vec![vec![0; m]; m+1];
        for i in 0..m{
            for j in 0..m{
                let mut start_idx = 5;
                loop{
                    let mut flg = true;
                    for x in (5-start_idx)..5{
                        if t[j][x-(5-start_idx)]!=t[i][x]{
                            flg = false;
                            break;
                        }
                    }
                    if flg{
                        break;
                    }
                    start_idx -= 1;
                    if start_idx==0{
                        break;
                    }
                }
                start_idxs[i][j] = start_idx;
            }
        }
        let mut dists = vec![vec![vec![]; m]; m+1];
        for last_idx in 0..m{
            for i in 0..m{
                let start_idx = start_idxs[last_idx][i];
                dists[last_idx][i] = vec![(0, 0); char2pos[t[last_idx].last().unwrap()].len()];
                for (cur_idx, cur_) in char2pos[t[last_idx].last().unwrap()].iter().enumerate(){
                    let mut cur = cur_.clone();
                    for j in start_idx..5{
                        let mut best_pos: Pos = Pos::new(0, 0);
                        let mut best_pos_idx = 0;
                        let mut min_dist = 1e9 as usize;
                        for (ii, p) in char2pos[&t[i][j]].iter().enumerate(){
                            let dist = cur.dist(*p);
                            if dist<min_dist{
                                min_dist = dist;
                                best_pos = *p;
                                best_pos_idx = ii;
                            }
                        }
                        cur = best_pos;
                        dists[last_idx][i][cur_idx] = (dists[last_idx][i][cur_idx].0+min_dist, best_pos_idx);
                    }
                }
            }
        }
        let last_idx = m;
        for i in 0..m{
            let start_idx = start_idxs[last_idx][i];
            dists[last_idx][i] = vec![(0, 0)];
            let mut cur = Pos::new(si, sj);
            for j in 0..5{
                let mut best_pos: Pos = Pos::new(0, 0);
                let mut best_pos_idx = 0;
                let mut min_dist = 1e9 as usize;
                for (ii, p) in char2pos[&t[i][j]].iter().enumerate(){
                    let dist = cur.dist(*p);
                    if dist<min_dist{
                        min_dist = dist;
                        best_pos = *p;
                        best_pos_idx = ii;
                    }
                }
                dists[last_idx][i][0] = (dists[last_idx][i][0].0+min_dist, best_pos_idx);
            }
        }


        Self { n, m, si, sj, pos2char: A, char2pos, t, start_idxs, dists }
    }
}

struct Neighbor{
    neighbor_type: usize,
    idxs: Vec<usize>,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    order: Vec<usize>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut appeared = HashSet::new();
        let mut order = vec![0];
        appeared.insert(0);
        while appeared.len()<input.m{
            let mut best_idx = 0;
            let mut best_score = 0;
            for i in 0..input.m{
                if appeared.contains(&i){
                    continue;
                }
                let mut score = 0;
                if input.start_idxs[*order.last().unwrap()][i]>=best_score{
                    best_score = score;
                    best_idx = i;
                }
            }
            order.push(best_idx);
            appeared.insert(best_idx);
        }
        Self {order}
    }

    fn init_state_random(input: &Input) -> Self{
        let mut rng = rand::thread_rng();
        let mut notappeared: HashSet<usize> = (0..input.m).collect();
        let mut order = vec![];
        while !notappeared.is_empty() {
            let idx = *notappeared.iter().choose(&mut rng).unwrap();
            notappeared.remove(&idx);
            order.push(idx);
        }
        Self {order}
    }

    fn update(&mut self, params: &Neighbor){
        if params.neighbor_type==0{
            self.order.swap(params.idxs[0], params.idxs[1]);
        } else if params.neighbor_type==1 {
            let idx = self.order.remove(params.idxs[0]);
            self.order.insert(params.idxs[1], idx);
        } else{
            let (l, r, to) = (params.idxs[0], params.idxs[1], params.idxs[2]);
            let mut new_order = Vec::with_capacity(self.order.len());
            if l<to{
                new_order.extend_from_slice(&self.order[..l]);
                new_order.extend_from_slice(&self.order[r..to]);
                new_order.extend_from_slice(&self.order[l..r]);
                new_order.extend_from_slice(&self.order[to..]);
            } else {
                new_order.extend_from_slice(&self.order[..to]);
                new_order.extend_from_slice(&self.order[l..r]);
                new_order.extend_from_slice(&self.order[to..l]);
                new_order.extend_from_slice(&self.order[r..]);
            }
            self.order = new_order;
        }
    }

    fn undo(&mut self, params: &Neighbor){
        if params.neighbor_type==0{
            self.order.swap(params.idxs[0], params.idxs[1]);
        } else if params.neighbor_type==1 {
            let idx = self.order.remove(params.idxs[1]);
            self.order.insert(params.idxs[0], idx);
        } else{
            let (l, r, to) = (params.idxs[0], params.idxs[1], params.idxs[2]);
            let mut new_order = Vec::with_capacity(self.order.len());
            if l<to{
                new_order.extend_from_slice(&self.order[..l]);
                new_order.extend_from_slice(&self.order[to..(to+(r-l))]);
                new_order.extend_from_slice(&self.order[l..to]);
                new_order.extend_from_slice(&self.order[(to+(r-l))..]);
            } else {
                new_order.extend_from_slice(&self.order[..to]);
                new_order.extend_from_slice(&self.order[to+(r-l)..l+(r-l)]);
                new_order.extend_from_slice(&self.order[to..to+(r-l)]);
                new_order.extend_from_slice(&self.order[r..]);
            }
            self.order = new_order;
        }
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        let mode_flg = rng.gen::<usize>()%70;
        if mode_flg<33{
            Neighbor{neighbor_type: 0, idxs: vec![rng.gen::<usize>()%input.m, rng.gen::<usize>()%input.m]}
        } else if mode_flg<66 {
            Neighbor{neighbor_type: 1,  idxs: vec![rng.gen::<usize>()%input.m, rng.gen::<usize>()%input.m]}
        } else{
            loop {
                let l = rng.gen::<usize>()%input.m;
                let r = rng.gen::<usize>()%input.m;
                let to = rng.gen::<usize>()%input.m;
                if (l<=to && to<r){
                    continue;
                }
                if !(l<r){
                    continue;
                }
                if input.m<=to+(r-l){
                    continue;
                }
                return Neighbor{neighbor_type: 2,  idxs: vec![l, r, to]}
            }
            Neighbor{neighbor_type: 2,  idxs: vec![rng.gen::<usize>()%input.m, rng.gen::<usize>()%input.m, rng.gen::<usize>()%input.m]}
        }
    }

    fn solve(&self, input: &Input) -> (Vec<Pos>, f64){
        // (今どこ, スコア, ans)
        let mut cur = vec![(Pos::new(input.si, input.sj), 0.0, vec![])];
        let mut last_idx = input.m;
        for i in self.order.iter(){
            let start_idx = input.start_idxs[last_idx][*i];
            for j in start_idx..5{
                let mut new_cur = vec![(Pos::new(0, 0), 99999999.0, vec![]); input.char2pos[&input.t[*i][j]].len()];
                for (p1, score, ans) in cur.iter(){
                    for (idx, p2) in input.char2pos[&input.t[*i][j]].iter().enumerate(){
                        let score_ = score+(p1.dist(*p2)+1) as f64;
                        if score_<new_cur[idx].1{
                            new_cur[idx] = (*p2, score_, ans.clone());
                            new_cur[idx].2.push(*p2);
                        }
                    }
                }
                cur = new_cur;
            }
            last_idx = *i;
        }
        let mut best_idx = 0;
        let mut best_score = 9999999.0;
        for (idx, (_, score, _)) in cur.iter().enumerate(){
            if *score<best_score{
                best_score = *score;
                best_idx = idx;
            }
        }
        (cur[best_idx].2.clone(), cur[best_idx].1)
    }

    fn solve_rough(&self, input: &Input) -> (Vec<Pos>, f64){
        // input.dists: (last_idx, new_idx, last_charpos_idx) -> (dist, charpos_idx)
        let mut cur = Pos::new(input.si, input.sj);
        let mut score = 0.0;
        let mut ans: Vec<Pos> = vec![];
        let mut last_idx = input.m;
        for i in self.order.iter(){
            let start_idx = input.start_idxs[last_idx][*i];
            for j in start_idx..5{
                let mut best_pos: Pos = Pos::new(0, 0);
                let mut min_dist = 1e9 as usize;
                for p in input.char2pos[&input.t[*i][j]].iter(){
                    let dist = cur.dist(*p);
                    if dist<min_dist{
                        min_dist = dist;
                        best_pos = *p;
                    }
                }
                cur = best_pos;
                ans.push(cur);
                score += (min_dist+1) as f64;
            }
            last_idx = *i;
        }
        (ans, score)
    }

    fn get_score(&self, input: &Input) -> f64{
        let mut cur = Pos::new(input.si, input.sj);
        let mut score = 0.0;
        let mut ans: Vec<Pos> = vec![];
        let mut last_idx = input.m;
        let mut last_charpos_idx = 0;
        for i in self.order.iter(){
            let (dist, charpos_idx) = input.dists[last_idx][*i][last_charpos_idx];
            score += dist as f64;
            last_idx = *i;
            last_charpos_idx = charpos_idx;
        }
        score
    }

    fn get_score_rough(&self, input: &Input) -> f64{
        self.solve_rough(input).1
    }

    fn print(&mut self, input: &Input){
        let ans = self.solve(input);
        for p in ans.0{
            println!("{} {}", p.row, p.col);
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
    let start_temp: f64 = 1.0;
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
        let score_diff = cur_score-new_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            accepted_cnt += 1;
            // eprintln!("{} {}", cur_score, new_score);
            cur_score = new_score;
            last_updated = all_iter;
            //state.print(input);
            if new_score<best_score{
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
    let mut best_state = simanneal(input, init_state, timer, 1.8);

    best_state.print(input);
}
