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

const INF: i64 = 1 << 31;
const DIR: [(i64, i64); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Board {
    M: Vec<Vec<bool>>,
    s: (usize, usize)
}

impl Board {
    pub fn from_str(m: Vec<Vec<char>>, s: (usize, usize)) -> Board{
        let N = m.len();
        let mut M = vec![vec![true; N+2]; N+2];
        for i in 0..N{
            for j in 0..N{
                if m[i][j] == '.'{
                    M[i+1][j+1] = false;
                }
            }
        }
        Self {M, s: (s.0+1, s.1+1)}
    }

    pub fn update(&mut self, add: &Vec<(usize, usize)>, remove: &Vec<(usize, usize)>){
        for (x, y) in add.iter(){
            self.M[*x+1][*y+1] = true;
        }
        for (x, y) in remove.iter(){
            self.M[*x+1][*y+1] = false;
        }
    }

    pub fn simulate_log(&self) -> Vec<bool>{
        let N = self.M.len();
        let mut appeared = vec![false; 4*N*N];
        let mut cur = self.s.0*4*N+self.s.1*4;
        let mut cnt = 0;
        while (!appeared[cur]){
            appeared[cur] = true;
            let xy = ((cur/(4*N)) as i64, ((cur/4)%N) as i64);
            let dir = cur%4;
            let xy_ = (xy.0+DIR[dir].0, xy.1+DIR[dir].1);
            if !self.M[xy_.0 as usize][xy_.1 as usize] {
                cur = (xy_.0 as usize)*4*N + (xy_.1 as usize)*4 + dir;
                cnt += 1;
            } else {
                cur = cur-dir + ((dir+1)%4);
            }
        }
        appeared
    }


    pub fn simulate(&self) -> usize{
        let N = self.M.len();
        let mut appeared = vec![false; 4*N*N];
        let mut cur = self.s.0*4*N+self.s.1*4;
        let mut cnt = 0;
        while (!appeared[cur]){
            appeared[cur] = true;
            let xy = ((cur/(4*N)) as i64, ((cur/4)%N) as i64);
            let dir = cur%4;
            let xy_ = (xy.0+DIR[dir].0, xy.1+DIR[dir].1);
            if !self.M[xy_.0 as usize][xy_.1 as usize] {
                cur = (xy_.0 as usize)*4*N + (xy_.1 as usize)*4 + dir;
                cnt += 1;
            } else {
                cur = cur-dir + ((dir+1)%4);
            }
        }
        cnt
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    N: usize,
    s: (usize, usize),
    board: Board,
}

impl Input {
    fn read_input() -> Self {
        input! {
            N: usize,
            sx: usize,
            sy: usize,
            m_str: [Chars; N],
        }
        let board = Board::from_str(m_str, (sx, sy));
        Self { N, s: (sx, sy), board}
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    board: Board
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut board = input.board.clone();
        let mut add = vec![];
        for i in 0..input.N{
            if i%2==0{
                add.push((0, i));
                add.push((input.N-1, i));
                add.push((i, 0));
                add.push((i, input.N-1));
            }
        }
        board.update(&add, &vec![]);
        Self {board}
    }

    fn update(&mut self, add: &Vec<(usize, usize)>, remove: &Vec<(usize, usize)>){
        self.board.update(add, remove);

    }

    fn undo(&mut self, add: &Vec<(usize, usize)>, remove: &Vec<(usize, usize)>){
        self.board.update(remove, add);
    }

    fn get_neighbor(&mut self, input: &Input) -> (Vec<(usize, usize)>, Vec<(usize, usize)>){
        let N = input.N;
        let mut rng = rand::thread_rng();
        let mode_flg = rng.gen::<usize>()%100;
        let mut ret = (vec![], vec![]);
        if mode_flg<999{
            let simulate_log = self.board.simulate_log();
            for i in 0..1000{
                let (x, y) = (rng.gen::<usize>()%N, rng.gen::<usize>()%N);
                let xy = (x+1)*4*N+(y+1)*4;
                if (simulate_log[xy] || simulate_log[xy+1] || simulate_log[xy+2] || simulate_log[xy+3]){
                // if !self.board.M[x+1][y+1]{
                    ret.0.push((x, y));
                    break;
                }
            }
        } else {
            for i in 0..1000{
                let (x, y) = (rng.gen::<usize>()%N, rng.gen::<usize>()%N);
                if self.board.M[x+1][y+1]&&(!input.board.M[x+1][y+1]){
                    ret.1.push((x, y));
                    break;
                }
            }
        }
        ret
    }

    fn get_score(&self) -> i64{
        let E = self.board.M.len() as i64;
        1000000*(self.board.simulate() as i64)/(4*E*E)
    }

    fn print(&mut self, input: &Input){
        let N = input.N;
        let mut diffs = vec![];
        for i in 0..N{
            for j in 0..N{
                if (self.board.M[i+1][j+1]!=input.board.M[i+1][j+1]){
                    diffs.push((i, j));
                }
            }
        }
        println!("{}", diffs.len());
        let dst: Vec<String> = diffs.iter().map(|x| format!("{} {}", x.0, x.1)).collect();
        println!("{}", dst.join(" "));
    }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.get_score();
    let mut cur_score = best_score;
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 100000.0;
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
        let (add, remove) = state.get_neighbor(input);
        state.update(&add, &remove);
        let new_score = state.get_score();
        let score_diff = new_score-cur_score;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if (score_diff>=0){// || ((score_diff as f64)/temp).exp() > rng.gen::<f64>()){
            accepted_cnt += 1;
            eprintln!("{} {} {:?} {:?}", cur_score, new_score, add, remove);
            cur_score = new_score;
            last_updated = all_iter;
            //state.print(input);
            if new_score>best_score{
                best_state = state.clone();
                best_score = new_score;
            }
        } else {
            state.undo(&add, &remove);
            // state.undo(&params);
            // state.undo(input, params);
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("score  : {}", best_state.get_score());
    eprintln!("");
    best_state
}


fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.8);
    // let mut best_state = State::init_state(&input);
    // for i in 0..1{
    //     let mut state = solve(&input, &timer, 5.5*((i+1) as f64));
    //     if state.get_score(1)<best_state.get_score(1){
    //         best_state = state;
    //     }
    // }
    // best_state.print();
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut state = State::init_state(input);
    let mut best_state = simanneal(input, state, timer, tl);
    best_state.print(input);
    // best_state.print();
    // eprintln!("{:?}", state);
    // let mut block1 = Block::single_cube(0, (1, 1, 1));
    // block1.update_cubes(&vec![(1, (1, 2, 1)), (1, (1, 2, 2)), (0, (1, 1, 1))]);
    // //block1.add_cubes(&vec![(1, 2, 1), (2, 1, 1), (1, 1, 2)]);
    // let mut block2 = Block::single_cube(0, (8, 8, 8));
    // block2.add_cubes(&vec![(8, 7, 8), (8, 8, 7), (7, 8, 8)]);
    // eprintln!("{:?} {}", block1, block1.serial);
    // eprintln!("{:?} {}", block2, block2.serial);
}

// fn solve(input: &Input, timer:&Instant, tl: f64) -> State{
//     let init_state = State::init_state(input);
//     let best_state = simanneal(input, init_state, &timer, tl);
//     //println!("{}", best_state);
//     eprintln!("{}", timer.elapsed().as_secs_f64());
//     best_state
// }
