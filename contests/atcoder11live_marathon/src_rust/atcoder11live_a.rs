#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use grid::{Coordinate, Map2d, ADJACENTS};
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
    n: usize,
    start: Coordinate,
    map: Map2d<bool>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            si: usize,
            sj: usize
        }
        let start = Coordinate::new(si, sj);
        let mut map = Map2d::new(vec![false; n * n], n);
        for row in 0..n {
            input! {
                s: Chars
            }
            for col in 0..n {
                let b = s[col] == '#';
                map[Coordinate::new(row, col)] = b;
            }
        }
        Self { n, start, map}
    }

    fn to_index(&self, c: Coordinate, dir: usize) -> usize {
        ((c.row * self.n) + c.col) * 4 + dir
    }
}

static mut SEEN: [u32; 10000] = [0; 10000];
static mut ITERATION: u32 = 0;


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    map: Map2d<bool>
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut map = input.map.clone();
        // let mut add = vec![];
        // for i in 0..input.N{
        //     if i%2==0{
        //         add.push((0, i));
        //         add.push((input.N-1, i));
        //         add.push((i, 0));
        //         add.push((i, input.N-1));
        //     }
        // }
        // board.update(&add, &vec![]);
        Self {map}
    }

    pub fn simulate(&self, input: &Input) -> usize{
        unsafe{
            ITERATION += 1;
            let n = input.n;
            let mut cur = input.start.clone();
            let mut dir: usize = 1;
            let mut cnt = 0;
            while (SEEN[cur.to_index(n)*4+dir]!=ITERATION){
                SEEN[cur.to_index(n)*4+dir] = ITERATION;
                let next = cur+ADJACENTS[dir];
                if !next.in_map(input.n) || self.map[next] {
                    dir = (dir+1)%4;
                } else {
                    cur = next;
                    cnt += 1;
                }
            }
            cnt
        }
    }

    fn update(&mut self, turns: &Vec<Coordinate>){
        for c in turns.iter(){
            self.map[c] ^= true;
        }
    }

    fn get_neighbor(&mut self, input: &Input) -> Vec<Coordinate>{
        let n = input.n;
        let mut rng = rand::thread_rng();
        // let mode_flg = rng.gen::<usize>()%100;
        let mut ret = vec![];
        let c = Coordinate::new(rng.gen_range(0, input.n), rng.gen_range(0, input.n));
        if input.map[c] || input.start==c{
            return vec![];
        }
        ret.push(c);
        ret
    }

    fn get_score(&self, input: &Input) -> f64{
        let E = self.map.width as f64;
        1000000.0*(self.simulate(input) as f64)/(4.0*E*E)
    }

    fn print(&mut self, input: &Input){
        let n = input.n;
        let mut diffs = vec![];
        for i in 0..n{
            for j in 0..n{
                let c = Coordinate::new(i, j);
                if (self.map[c]!=input.map[c]){
                    diffs.push(c);
                }
            }
        }
        println!("{}", diffs.len());
        let dst: Vec<String> = diffs.iter().map(|c| format!("{} {}", c.row, c.col)).collect();
        println!("{}", dst.join("\n"));
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
        let turns = state.get_neighbor(input);
        state.update(&turns);
        let new_score = state.get_score(input);
        let score_diff = new_score-cur_score;
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
            state.update(&turns);
            // state.undo(&params);
            // state.undo(input, params);
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
    let mut best_state = state.clone();
    for i in 1..6{
        let mut tmp_state = simanneal(input, state.clone(), timer, 0.2*(i as f64));
        // eprintln!("{}", tmp_state.get_score(input));
        if tmp_state.get_score(input)>best_state.get_score(input){
            best_state = tmp_state;
        }
    }
    let mut best_state2 = simanneal(input, best_state, timer, tl);

    best_state2.print(input);
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


#[allow(dead_code)]
mod grid {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Coordinate {
        pub row: usize,
        pub col: usize,
    }

    impl Coordinate {
        pub const fn new(row: usize, col: usize) -> Self {
            Self { row, col }
        }

        pub fn in_map(&self, size: usize) -> bool {
            self.row < size && self.col < size
        }

        pub const fn to_index(&self, size: usize) -> usize {
            self.row * size + self.col
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

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        pub width: usize,
        pub height: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, width: usize) -> Self {
            let height = map.len() / width;
            debug_assert!(width * height == map.len());
            Self { width, height, map }
        }
    }

    impl<T> std::ops::Index<Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coordinate) -> &Self::Output {
            &self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::Index<&Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coordinate) -> &Self::Output {
            &self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<&Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &self.map[begin..end]
        }
    }

    impl<T> std::ops::IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &mut self.map[begin..end]
        }
    }
}
