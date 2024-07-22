#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
use itertools::Itertools;
use num::{range, ToPrimitive};
use num_integer::Roots;
use proconio::{input, marker::Chars};
use rand_core::{block, le};
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fmt::Display;
use std::cmp::{Reverse, Ordering};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;
use ndarray::prelude::*;
use num::BigUint;
use grid::{Coordinate, Map2d, ADJACENTS, CoordinateDiff, SIZE};

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    h: Map2d<i32>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            h_: [i32; 400]
            //m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let h = Map2d::new(h_);
        Self { n, h }
    }
}

struct Neighbor{
    mode: usize,
    idx1: usize,
    idx2: usize,
    idx3: usize,
    pre_score: Option<f64>,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    order: Vec<Coordinate>,
    score: Option<f64>
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut order = vec![];
        for i in 0..input.n{
            if i%2==0{
                for j in 0..input.n{
                    order.push(Coordinate::new(i, j));
                }
            } else {
                for j in (0..input.n).rev(){
                    order.push(Coordinate::new(i, j));
                }
            }
        }
        for i in 0..input.n{
            for j in 0..input.n{
                if i%2==0{
                    for j in 0..input.n{
                        if input.h[i][j]<0{
                            order.push(Coordinate::new(i, j));
                        }
                    }
                } else {
                    for j in (0..input.n).rev(){
                        if input.h[i][j]<0{
                            order.push(Coordinate::new(i, j));
                        }
                    }
                }
            }
        }
        Self {order, score: None}
    }

    fn update(&mut self, params: &Neighbor){
        if params.mode==0{
            // swap
            self.order.swap(params.idx1, params.idx2);
        } else if params.mode==1{
            // insert
            let tmp = self.order.remove(params.idx1);
            self.order.insert(params.idx2, tmp);
        } else if params.mode==2{
            // 2-opt
            self.order[params.idx1..params.idx2].reverse();
        } else{
            // 範囲insert
            let tmp = self.order.drain(params.idx1..params.idx2).collect::<Vec<_>>();
            self.order.splice(params.idx3..params.idx3, tmp);
        }
        self.score = None;
    }

    fn undo(&mut self, params: &Neighbor){
        if params.mode==0{
            // swap
            self.order.swap(params.idx1, params.idx2);
        } else if params.mode==1{
            let tmp = self.order.remove(params.idx2);
            self.order.insert(params.idx1, tmp);
        } else if params.mode==2{
            // 2-opt
            self.order[params.idx1..params.idx2].reverse();
        } else{
            // 範囲insert
            let idx1 = params.idx3;
            let idx2 = params.idx3+params.idx2-params.idx1;
            let idx3 = params.idx1;
            let tmp = self.order.drain(idx1..idx2).collect::<Vec<_>>();
            self.order.splice(idx3..idx3, tmp);
        }
        self.score = params.pre_score;
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let n = input.n;
        let mut rng = rand::thread_rng();
        let mode_flg = rng.gen::<usize>()%100;
        let mut mode: usize;
        let mut idx1: usize;
        let mut idx2: usize;
        let mut idx3: usize;
        if mode_flg<30{
            // swap
            if mode_flg<15{
                mode = 0;
            } else {
                mode = 1;
            }
            idx1 = rng.gen::<usize>()%self.order.len();
            idx2 = rng.gen::<usize>()%self.order.len();
            idx3 = rng.gen::<usize>()%self.order.len();
            while idx1==idx2{
                idx1 = rng.gen::<usize>()%self.order.len();
                idx2 = rng.gen::<usize>()%self.order.len();
            }
        } else if mode_flg < 70 {
            mode = 2;
            idx1 = rng.gen::<usize>()%self.order.len();
            idx2 = rng.gen::<usize>()%self.order.len();
            idx3 = rng.gen::<usize>()%self.order.len();
            while idx1>=idx2{
                idx1 = rng.gen::<usize>()%self.order.len();
                idx2 = rng.gen::<usize>()%self.order.len();
            }
        } else {
            mode = 3;
            idx1 = rng.gen::<usize>()%self.order.len();
            idx2 = rng.gen::<usize>()%self.order.len();
            idx3 = rng.gen::<usize>()%self.order.len();
            while idx1>=idx2 || idx3>=self.order.len()-(idx2-idx1) {
                idx1 = rng.gen::<usize>()%self.order.len();
                idx2 = rng.gen::<usize>()%self.order.len();
                idx3 = rng.gen::<usize>()%self.order.len();
            }

        }
        Neighbor{mode, idx1, idx2, idx3, pre_score: self.score}
    }

    fn get_score(&mut self, input: &Input) -> f64{
        if let Some(score) = self.score{
            return score;
        }
        let mut h = input.h.clone();
        let mut cur_c = Coordinate::new(0, 0);
        let mut cur_d: i32 = 0;
        let mut score = 0.0;
        for c in &self.order{
            let d = h[c];
            if d==0{
                continue;
            }
            let move_cost = cur_c.dist(c)*(100+cur_d as usize);
            score += move_cost as f64;
            if d>0 || cur_d+d>=0{
                cur_d += d;
                h[c] = 0;
            } else if d==input.h[c]{
                h[c] += cur_d;
                cur_d = 0;
            } else {
                score = f64::MAX;
                break;
            }
            cur_c = *c;
        }
        self.score = Some(score);
        score
    }

    fn print(&mut self, input: &Input){
        let mut cur_c = Coordinate::new(0, 0);
        let mut cur_d: i32 = 0;
        let mut h = input.h.clone();
        for c in &self.order{
            let d = h[c];
            if d==0{
                continue;
            }
            while c.row>cur_c.row{
                println!("D");
                cur_c.row += 1;
            }
            while c.row<cur_c.row{
                println!("U");
                cur_c.row -= 1;
            }
            while c.col>cur_c.col{
                println!("R");
                cur_c.col += 1;
            }
            while c.col<cur_c.col{
                println!("L");
                cur_c.col -= 1;
            }
            if d>0{
                println!("+{}", d);
                cur_d += d;
                h[c] = 0;
            } else if cur_d+d>=0{
                cur_d += d;
                println!("{}", d);
                h[c] = 0;
            } else if cur_d>0 {
                println!("-{}", cur_d);
                h[c] += cur_d;
                cur_d = 0;
            }
            cur_c = *c;
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
        let score_diff = cur_score-new_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            accepted_cnt += 1;
            eprintln!("{} {} {:?}", all_iter, cur_score, new_score);
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
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
}

#[allow(dead_code)]
mod grid {
    pub const SIZE: usize = 20;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Coordinate {
        pub row: usize,
        pub col: usize,
    }

    impl Coordinate {
        pub const fn new(row: usize, col: usize) -> Self {
            Self { row, col }
        }

        pub fn in_map(&self) -> bool {
            self.row < SIZE && self.col < SIZE
        }

        pub const fn to_index(&self) -> usize {
            self.row * SIZE + self.col
        }

        pub const fn dist(&self, other: &Self) -> usize {
            Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
        }

        const fn dist_1d(x0: usize, x1: usize) -> usize {
            (x0 as i64 - x1 as i64).abs() as usize
        }

    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CoordinateDiff {
        pub dr: usize,
        pub dc: usize,
    }

    impl CoordinateDiff {
        pub const fn new(dr: usize, dc: usize) -> Self {
            Self { dr, dc }
        }

        pub const fn invert(&self) -> Self {
            Self::new(0usize.wrapping_sub(self.dr), 0usize.wrapping_sub(self.dc))
        }
    }

    impl std::ops::Add<CoordinateDiff> for Coordinate {
        type Output = Coordinate;

        fn add(self, rhs: CoordinateDiff) -> Self::Output {
            Coordinate::new(self.row.wrapping_add(rhs.dr), self.col.wrapping_add(rhs.dc))
        }
    }

    pub const ADJACENTS: [CoordinateDiff; 4] = [
        CoordinateDiff::new(!0, 0),
        CoordinateDiff::new(0, 1),
        CoordinateDiff::new(1, 0),
        CoordinateDiff::new(0, !0),
    ];

    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    #[derive(Debug, Clone, Hash)]
    pub struct Map2d<T> {
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>) -> Self {
            Self { map }
        }
    }

    impl<T> std::ops::Index<Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coordinate) -> &Self::Output {
            &self.map[coordinate.row * SIZE + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * SIZE + coordinate.col]
        }
    }

    impl<T> std::ops::Index<&Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coordinate) -> &Self::Output {
            &self.map[coordinate.row * SIZE + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<&Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * SIZE + coordinate.col]
        }
    }

    impl<T> std::ops::Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * SIZE;
            let end = begin + SIZE;
            &self.map[begin..end]
        }
    }

    impl<T> std::ops::IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * SIZE;
            let end = begin + SIZE;
            &mut self.map[begin..end]
        }
    }
}
