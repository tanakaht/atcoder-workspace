#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
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


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    B: Vec<usize>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            B: [usize; 465],
            //m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        Self { B }
    }

    fn i2xy(i: usize) -> (usize, usize) {
        let mut x = (2*i).sqrt();
        if x*(x+1)/2>i{
            x -= 1;
        }
        (x, i-x*(x+1)/2)
    }

    fn xy2i(xy: (usize, usize)) -> usize{
        xy.0*(xy.0+1)/2 + xy.1
    }

    fn neighbor(i: usize) -> Vec<usize>{
        let mut ret = vec![];
        let (x, y) = Self::i2xy(i);
        for (x_, y_) in [(x-1, y-1), (x-1, y), (x, y-1), (x, y+1), (x+1,y), (x+1, y+1)].iter(){
            if *x_<30 && *y_<30{
                ret.push(Self::xy2i((*x_, *y_)));
            }
        }
        ret
    }

    fn is_children(i: usize, j: usize) -> bool{
        true
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Neighbor{
    flg: usize,
    i: usize,
    j: usize,
    idx: usize,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    B_init: Vec<usize>,
    B_cur: Vec<usize>,
    swaps: Vec<(usize, usize)>,
    score: f64,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut ret = Self {B_init: input.B.clone(), B_cur: input.B.clone(), swaps: vec![], score: 0.0};
        ret.score = ret.get_score(input);
        ret
    }

    fn simulate(&mut self){
        self.B_cur = self.B_init.clone();
        for (i, j) in self.swaps.iter(){
            (self.B_cur[*i], self.B_cur[*j]) = (self.B_cur[*j], self.B_cur[*i]);
        }
    }

    fn update(&mut self, params: &Neighbor){
        if params.flg==0{
            let (bi, bj) = (self.B_cur[params.i], self.B_cur[params.j]);
            self.B_cur[params.i] = bj;
            self.B_cur[params.j] = bi;
            self.swaps.push((params.i, params.j));
        } else {
            self.swaps.remove(params.idx);
            self.simulate();
        }
    }

    fn undo(&mut self, params: &Neighbor){
        if params.flg==0{
            let (bi, bj) = (self.B_cur[params.i], self.B_cur[params.j]);
            self.B_cur[params.i] = bj;
            self.B_cur[params.j] = bi;
            self.swaps.remove(self.swaps.len()-1);
        } else {
            self.swaps.insert(params.idx, (params.i, params.j));
            self.simulate();
        }
    }

    fn get_neighbor(&mut self, input: &Input, mode_flg: usize) -> Neighbor{
        let mut rng = rand::thread_rng();
        // let mode_flg = rng.gen::<usize>()%100;
        if mode_flg<50{
            let i = rng.gen::<usize>()%465;
            let j = *Input::neighbor(i).choose(&mut rng).unwrap();
            return Neighbor{flg: 0, i, j, idx: 0};
        } else {
            let idx = rng.gen::<usize>()%self.swaps.len();
            return Neighbor{flg: 0, i: self.swaps[idx].0, j: self.swaps[idx].1, idx};
        }
    }

    fn get_score(&self, input: &Input) -> f64{
        let mut E = 0;
        for i in 0..465{
            for j in Input::neighbor(i).iter(){
                if (i<*j && self.B_cur[i]>self.B_cur[*j]){
                    E += 1;
                }
            }
        }
        if self.swaps.len()>10000 {
            return 0.0;
        } else if E == 0{
            return 100000.0-(5*self.swaps.len()) as f64;
        } else {
            return 50000.0-(50*E) as f64;
        }
    }

    fn print(&mut self, input: &Input){
        println!("{}", self.swaps.len());
        for (i, j) in self.swaps.iter(){
            let xyi = Input::i2xy(*i);
            let xyj = Input::i2xy(*j);
            println!("{} {} {} {}", xyi.0, xyi.1, xyj.0, xyj.1);
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
        let neighbor = state.get_neighbor(input, 0);
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
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
}
