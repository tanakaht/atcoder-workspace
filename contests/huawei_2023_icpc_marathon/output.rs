#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
#[allow(unused_imports)]
use rand::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fmt::Display;
use std::io;
use std::process::exit;
use std::str::FromStr;
use std::time::Instant;
#[allow(dead_code)]
fn read_line() -> String {
    let mut buffer = String::new();
    io::stdin()
        .read_line(&mut buffer)
        .expect("failed to read line");
    buffer
}
#[allow(dead_code)]
fn read<T: FromStr>() -> Result<T, T::Err> {
    read_line().trim().parse::<T>()
}
#[allow(dead_code)]
fn read_vec<T: FromStr>() -> Result<Vec<T>, T::Err> {
    read_line()
        .split_whitespace()
        .map(|x| x.parse::<T>())
        .collect()
}
#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    X: Vec<f64>,
}
impl Input {
    fn read_input() -> Self {
        let eles = read_vec::<f64>().unwrap();
        let n = eles[0] as usize;
        let X = eles[1..].to_vec();
        Self { n, X }
    }
}
struct Neighbor {
    x: usize,
}
#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State {
    x: usize,
}
impl State {
    fn init_state(input: &Input) -> Self {
        Self { x: 1 }
    }
    fn update(&mut self, params: &Neighbor) {}
    fn undo(&mut self, params: &Neighbor) {}
    fn get_neighbor(&mut self, input: &Input) -> Neighbor {
        let n = input.n;
        let mut rng = rand::thread_rng();
        Neighbor {
            x: rng.gen::<usize>() % 100,
        }
    }
    fn get_score(&self, input: &Input) -> f64 {
        1.0
    }
    fn print(&mut self, input: &Input) {
        let n = input.n;
    }
}
fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State {
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
        if elasped_time >= limit {
            break;
        }
        all_iter += 1;
        let neighbor = state.get_neighbor(input);
        state.update(&neighbor);
        let new_score = state.get_score(input);
        let score_diff = new_score - cur_score;
        temp = start_temp + (end_temp - start_temp) * elasped_time / limit;
        if score_diff >= 0.0 || rng.gen_bool((score_diff as f64 / temp).exp()) {
            accepted_cnt += 1;
            cur_score = new_score;
            last_updated = all_iter;
            if new_score > best_score {
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
    eprintln!("{:?}", input);
    solve(&input, &timer, 1.8);
}
fn solve(input: &Input, timer: &Instant, tl: f64) {
    let mut init_state = State::init_state(input);
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
}

