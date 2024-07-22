#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap, VecDeque};
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


#[allow(unused_macros)]
macro_rules! min {
    ($a:expr $(,)*) => {{
        $a
    }};
    ($a:expr, $b:expr $(,)*) => {{
        std::cmp::min($a, $b)
    }};
    ($a:expr, $($rest:expr),+ $(,)*) => {{
        std::cmp::min($a, min!($($rest),+))
    }};
}

#[allow(unused_macros)]
macro_rules! max {
    ($a:expr $(,)*) => {{
        $a
    }};
    ($a:expr, $b:expr $(,)*) => {{
        std::cmp::max($a, $b)
    }};
    ($a:expr, $($rest:expr),+ $(,)*) => {{
        std::cmp::max($a, max!($($rest),+))
    }};
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    t: usize,
    h: usize,
    w: usize,
    i0: usize,
    g: Vec<Vec<usize>>,
    k: usize,
    sd: Vec<(usize, usize)>,
    flgs: Vec<Flg>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            t: usize,
            h: usize,
            w: usize,
            i0: usize,
            H: [String; h-1],
            V: [String; h],
            k: usize,
            sd_: [(usize, usize); k],
        }
        let mut g: Vec<Vec<usize>> = vec![vec![]; h*w];
        let sd = sd_.iter().map(|&(s, d)| ((s-1)*2, (d-1)*2+1)).collect_vec();
        let flgs = sd_.iter().map(|&(s, d)| Flg::from_invalid((s-1)*2, (d-1)*2+1)).collect_vec();
        for i in 0..(h-1){
            let H_ = usize::from_str_radix(&H[i], 2).unwrap();
            for j in 0..w{
                if (H_>>(w-1-j))&1==0{
                    g[i*w+j].push((i+1)*w+j);
                    g[(i+1)*w+j].push(i*w+j);
                }
            }
        }
        for i in 0..h{
            let V_ = usize::from_str_radix(&V[i], 2).unwrap();
            for j in 0..(w-1){
                if (V_>>(w-2-j))&1==0{
                    g[i*w+j].push(i*w+j+1);
                    g[i*w+j+1].push(i*w+j);
                }
            }
        }
        Self { t, h, w, i0, g, k, sd, flgs }
    }
}



#[allow(unused_variables)]
#[derive(Debug, Clone, Eq, PartialEq)]
struct Flg{
    bit0: u128,
    bit1: u128,
}

impl Flg{
    fn one() -> Self{
        Self{bit0: (1<<100)-1, bit1: (1<<100)-1}
    }

    fn zero() -> Self{
        Self{bit0: 0, bit1: 0}
    }

    fn popcnt(&self) -> usize{
        self.bit0.count_ones() as usize + self.bit1.count_ones() as usize
    }

    // fn from_valids(l: usize, r:usize) -> Self{
    // }

    // (l, r)がinvalid
    fn from_invalid(l: usize, r:usize) -> Self{
        if l<100{
            if r<100{
                Self{bit0: ((1<<100)-1)^((1<<r)-1)^((1<<(l+1))-1), bit1: (1<<100)-1}
            } else {
                Self{bit0: (1<<(l+1))-1, bit1: ((1<<100)-1)^((1<<(r-100))-1)}
            }
        } else {
            Self{bit0: (1<<100)-1, bit1: ((1<<100)-1)^((1<<(r-100))-1)^((1<<(l-100+1))-1)}
        }
    }

    // [l, r)がinvalid
    fn from_invalid2(l: usize, r:usize) -> Self{
        if l<100{
            if r<100{
                Self{bit0: ((1<<100)-1)^((1<<r)-1)^((1<<(l))-1), bit1: (1<<100)-1}
            } else {
                Self{bit0: (1<<(l))-1, bit1: ((1<<100)-1)^((1<<(r-100))-1)}
            }
        } else {
            Self{bit0: (1<<100)-1, bit1: ((1<<100)-1)^((1<<(r-100))-1)^((1<<(l-100))-1)}
        }
    }

    fn is_ok(&self, i: usize)->bool{
        if i<100{
            (self.bit0>>i)&1==1
        } else {
            (self.bit1>>(i-100))&1==1
        }
    }

    fn or(&self, other: &Flg) -> Flg{
        Self { bit0: self.bit0|other.bit0, bit1: self.bit1|other.bit1 }
    }

    fn and(&self, other: &Flg) -> Flg{
        Self { bit0: self.bit0&other.bit0, bit1: self.bit1&other.bit1 }
    }

    fn eq(&self, other: &Flg) -> bool{
        self.bit0==other.bit0 && self.bit1==other.bit1
    }

    fn is_bigger(&self, other: &Flg) -> bool{
        (self.bit0&other.bit0)>=self.bit0 && (self.bit1&other.bit1)>=self.bit1
    }

    // fn min_bit(&self, from: usize) -> usize{
    // }

    fn max_bit(&self, from: usize) -> Option<usize>{
        let flg = if (from<100){ self.and(&Self{bit0: (1<<(from+1))-1, bit1: 0}) } else { self.and(&Self{bit0: (1<<100)-1, bit1: (1<<(from-100+1))-1}) };
        if flg.bit1>0{
            let bit = flg.bit1;
            let mut l= 0;
            let mut r= 100;
            // [l, r)にbitが立っている
            while r-l>1{
                let m = (l+r)/2;
                if bit&((1<<m)-1)==bit{
                    r = m;
                } else {
                    l = m;
                }
            }
            Some(l+100)
        } else if flg.bit0>0{
            let bit = flg.bit0;
            let mut l= 0;
            let mut r= 100;
            // [l, r)にbitが立っている
            while r-l>1{
                let m = (l+r)/2;
                if bit&((1<<m)-1)==bit{
                    r = m;
                } else {
                    l = m;
                }
            }
            Some(l)
        } else {
            None
        }
    }
}


impl PartialOrd for Flg {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.popcnt().cmp(&other.popcnt()))
    }
}


impl Ord for Flg {
    fn cmp(&self, other: &Self) -> Ordering {
        self.popcnt().cmp(&other.popcnt())
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct NeighborParam{
    moves: Vec<(usize, usize, usize)> // i番目の作物をjからkに移動
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    places: Vec<usize>,
    place2idx: Vec<HashSet<usize>>,
    place2score: Vec<usize>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut place2idx = vec![HashSet::new(); input.h*input.w+1];
        for i in 0..input.k{
            place2idx[input.h*input.w].insert(i);
        }
        let mut state = Self{places: vec![input.h*input.w; input.k], place2idx, place2score: vec![0; input.h*input.w]};
        // moveを作る
        // dfsord
        let mut dfs_ord = vec![];
        let mut q = vec![];
        let mut appeared = vec![false; input.h*input.w];
        q.push(input.i0*input.w);
        while !q.is_empty(){
            let idx = q.pop().unwrap();
            if appeared[idx]{
                continue;
            }
            appeared[idx] = true;
            dfs_ord.push(idx);
            for idx2 in input.g[idx].iter(){
                if !appeared[*idx2]{
                    q.push(*idx2);
                }
            }
        }
        // 詰め込む
        let mut moves = vec![];
        let mut isd = input.sd.iter().enumerate().map(|(i, &(s, d))| (i, s, d)).collect_vec();
        isd.sort_by_key(|&(_, s, d)| d);
        let mut last_d = 0;
        while !isd.is_empty(){
            for (idx_isd, idx) in dfs_ord.iter().enumerate(){
                if let Some(&(i, s, d)) = isd.get(idx_isd){
                    moves.push((i, input.h*input.w, *idx));
                    last_d = max!(last_d, d);
                } else {
                    last_d = 3000;
                    break;
                }
            }
            isd = isd.iter().filter(|&(_, s, d)| *s>last_d).map(|&(i, s, d)| (i, s, d)).collect_vec();
            isd.sort_by_key(|&(_, s, d)| d);
        }
        let neighborparam = NeighborParam{moves};
        state.update(&neighborparam);
        state
    }

    fn get_mapflg(&self, input: &Input, idx: usize) -> Flg{
        self.place2idx[idx].iter().fold(Flg::one(), |acc, &i| acc.and(&input.flgs[i]))
    }

    fn get_reachmap(&self, input: &Input) -> Vec<Flg>{
        let mut flgs = vec![Flg::zero(); input.h*input.w];  // 入った時に更新
        let mut last_updated = vec![Flg::zero(); input.h*input.w];  // q入れた時に更新
        // let mut q = VecDeque::new();
        let mut q: BinaryHeap<(Flg, usize)> = BinaryHeap::new();
        q.push((self.get_mapflg(input, input.i0*input.w), input.i0*input.w));
        while !q.is_empty(){
            let (flg, idx) = q.pop().unwrap();
            flg.popcnt();
            if flg.eq(&flgs[idx]) || !flgs[idx].is_bigger(&flg){
                continue;
            }
            flgs[idx] = flg.clone();
            for idx2 in input.g[idx].iter(){
                let flg2 = flg.and(&self.get_mapflg(input, *idx2)).or(&flgs[*idx2]).or(&last_updated[*idx2]);
                if !flg2.eq(&last_updated[*idx2]) && last_updated[*idx2].is_bigger(&flg2){
                    last_updated[*idx2] = flg2.clone();
                    q.push((flg2, *idx2));
                }
            }
        }
        flgs
    }


    fn is_ok(&self, input: &Input) -> bool{
        for (i, idxs_) in self.place2idx.iter().enumerate(){
            if i==input.h*input.w{
                continue;
            }
            let mut idxs = idxs_.iter().map(|&c| c).collect_vec();
            idxs.sort_by_key(|&i_| input.sd[i_].1);
            let mut flg = Flg::one();
            let mut last_d = 0;
            for idx in idxs.iter(){
                let (s, d) = input.sd[*idx];
                flg = flg.and(&input.flgs[*idx]);
                if let Some(s_) = flg.max_bit(s){
                    if s_<last_d{
                        return false;
                    }
                    last_d = d;
                } else {
                    return false;
                }
            }
        }

        let mut flgs = self.get_reachmap(input);
        for (i, flg_) in flgs.iter().enumerate(){
            let mut idxs = self.place2idx[i].iter().map(|&c| c).collect_vec();
            let mut flg = flg_.clone();
            idxs.sort_by_key(|&i_| input.sd[i_].1);
            for idx in idxs.iter(){
                let (s, d) = input.sd[*idx];
                if flg.max_bit(s).is_none() || !flg.is_ok(d){
                    return false;
                }
                flg = flg.and(&Flg::from_invalid2(0, d));
            }
        }
        true
    }

    fn update(&mut self, params: &NeighborParam){
        for (i, j, k) in params.moves.iter(){
            self.places[*i] = *k;
            self.place2idx[*j].remove(i);
            self.place2idx[*k].insert(*i);
        }
    }

    fn undo(&mut self, params: &NeighborParam){
        for (i, j, k) in params.moves.iter(){
            self.places[*i] = *j;
            self.place2idx[*k].remove(i);
            self.place2idx[*j].insert(*i);
        }
    }

    fn get_neighbor(&mut self, input: &Input) -> NeighborParam{
        let mut rng = rand::thread_rng();
        let mode_flg = rng.gen::<usize>()%100;
        let mut moves = vec![];
        if mode_flg<60{
            //move
            for _ in 0..100{
                let i = **self.place2idx[input.h*input.w].iter().collect_vec().choose(&mut rng).unwrap();
                let k = rng.gen::<usize>()%(input.h*input.w);
                if (self.places[i]!=k){
                    moves.push((i, self.places[i], k));
                    break;
                }
            }
        } else if mode_flg<80 {
            //move
            for _ in 0..100{
                let i = rng.gen::<usize>()%input.k;
                let k = rng.gen::<usize>()%(input.h*input.w+1);
                if (self.places[i]!=k){
                    moves.push((i, self.places[i], rng.gen::<usize>()%(input.h*input.w)));
                    break;
                }
            }
        } else {
            // swap
            for _ in 0..100{
                let i1 = rng.gen::<usize>()%input.k;
                let i2 = rng.gen::<usize>()%input.k;
                let k1 = self.places[i1];
                let k2 = self.places[i2];
                if (k1!=k2){
                    moves.push((i1, k1, k2));
                    moves.push((i2, k2, k1));
                    break;
                }
            }
        }
        NeighborParam{moves}
    }

    fn get_score(&self, input: &Input, w: f64) -> f64{
        for (i, idxs_) in self.place2idx.iter().enumerate(){
            if i==input.h*input.w{
                continue;
            }
            let mut idxs = idxs_.iter().map(|&c| c).collect_vec();
            idxs.sort_by_key(|&i_| input.sd[i_].1);
            let mut flg = Flg::one();
            let mut last_d = 0;
            for idx in idxs.iter(){
                let (s, d) = input.sd[*idx];
                flg = flg.and(&input.flgs[*idx]);
                if let Some(s_) = flg.max_bit(s){
                    if s_<last_d{
                        return -10000000.0;
                    }
                    last_d = d;
                } else {
                    return -10000000.0;
                }
            }
        }

        let mut valid_x = 0;
        let mut invalid_x = 0;
        let mut map_x = 0;

        let mut flgs = self.get_reachmap(input);
        for (i, flg_) in flgs.iter().enumerate(){
            let mut idxs = self.place2idx[i].iter().map(|&c| c).collect_vec();
            let mut flg = flg_.clone();
            idxs.sort_by_key(|&i_| input.sd[i_].1);
            for idx in idxs.iter(){
                let (s, d) = input.sd[*idx];
                if !flg.is_ok(d){
                    return -10000000.0;
                }
                if let Some(s_) = flg.max_bit(s){
                    flg = flg.and(&Flg::from_invalid2(0, d));
                    valid_x += d-s+1;
                    invalid_x += s-s_;
                    map_x += (d-s_+1)*self.place2score[i];
                } else {
                    return -10000000.0;
                }
            }
        }
        (valid_x as f64)*3.0 - (invalid_x as f64)*10.0*w - (map_x as f64)*0.0
        // valid_x as f64
    }

    fn get_raw_score(&self, input: &Input) -> f64{
        if self.is_ok(input){
            let mut score = 0;
            for (i, place) in self.places.iter().enumerate(){
                if *place<input.h*input.w{
                    score += (input.sd[i].1-input.sd[i].0+1)/2;
                }
            }
            (score as f64)*1000000.0/((input.h*input.w*input.t) as f64)
        } else {
            -10000000.0
        }
    }

    fn print(&mut self, input: &Input){
        // is_ok前提
        let mut anss = vec![];
        let mut flgs = self.get_reachmap(input);
        for (i, flg) in flgs.iter().enumerate(){
            for idx in self.place2idx[i].iter(){
                let (s, d) = input.sd[*idx];
                let maxbit = flg.max_bit(s).unwrap();
                anss.push((idx+1, i/input.w, i%input.w ,maxbit/2+1));
            }
        }
        println!("{}", anss.len());
        for (k, i, j, s) in anss.iter(){
            println!("{} {} {} {}", k, i, j, s);
        }
    }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.get_score(input, 1.0);
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
        let new_score = state.get_score(input, 1.0-elasped_time/limit);
        let score_diff = new_score-cur_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            eprintln!("{} {} {:?}", new_score, all_iter, neighbor);
            accepted_cnt += 1;
            // eprintln!("{} {} {:?}", cur_score, new_score, all_iter);
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
    eprintln!("score  : {}", best_state.get_score(input, 0.0));
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
    // eprintln!("{}", init_state.get_score(input));
    //init_state.print(input);
    // eprintln!("{:?}", init_state.get_mapflg(input, input.i0*input.w));
    // eprintln!("{:?}", init_state.get_reachmap(input)[input.i0*input.w]);
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
    // eprintln!("{}", best_state.is_ok(input));
    // let reachmap = best_state.get_reachmap(input);
    // eprintln!("{:b} {:b}", reachmap[input.i0*input.w].bit0, reachmap[input.i0*input.w].bit1);
    // // for t in 0..200{
    //     let mut tmp = vec![vec![]; 20];
    //     for i in 0..20{
    //         for j in 0..20{
    //             tmp[i].push(reachmap[i*input.w+j].is_ok(t));
    //         }
    //     }
    // }
}
