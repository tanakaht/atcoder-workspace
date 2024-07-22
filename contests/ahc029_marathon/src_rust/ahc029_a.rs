#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
use itertools::Itertools;
use num::{range, ToPrimitive};
use num_integer::Roots;
use proconio::{input, marker::Chars, source::line::LineSource};
use std::io::{stdin, stdout, BufReader, Write};
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
    m: usize,
    k: usize,
    t: usize,
    init_projects: Vec<(i64, usize)>,
    init_hands: Vec<(usize, usize)>,
}

impl Input {
    fn read_input() -> Self {
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            n: usize,
            m: usize,
            k: usize,
            t: usize,
            init_hands: [(usize, usize); n], // a is Vec<i32>, n-array.
            init_projects: [(i64, usize); m], // a is Vec<i32>, n-array.
        }
        Self { n, m, k, t, init_projects, init_hands }
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    n: usize,
    m: usize,
    k: usize,
    max_turn: usize,
    projects: Vec<Option<(i64, usize)>>,
    n_projects: usize,
    hands: Vec<Option<(usize, usize)>>,
    cur_turn: usize,
    money: i64,
    l: usize,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let n = input.n;
        let m = input.m;
        let k = input.k;
        let max_turn = input.t;
        let projects = input.init_projects.iter().map(|&c| Some(c)).collect_vec();
        let hands = input.init_hands.iter().map(|&c| Some((c.0, c.1))).collect_vec();
        Self{n, m, k, max_turn, projects, n_projects: m, hands, cur_turn: 0, money:0, l: 0}
    }

    fn update_use_card(&mut self, c: usize, m: usize){
        let (t, w) = self.hands[c].unwrap();
        if t==0{
            self.projects[m] = self.projects[m].map(|(h, v)| (h-w as i64, v));
            if self.projects[m].unwrap().0<=0{
                self.money += self.projects[m].unwrap().1 as i64;
                self.projects[m] = None;
                self.n_projects -= 1;
            }
        } else if t==1{
            self.projects = self.projects.iter().map(|&c| c.map(|(h, v)| (h-w as i64, v))).collect_vec();
            for i in 0..self.projects.len(){
                if self.projects[i].unwrap().0<=0{
                    self.money += self.projects[i].unwrap().1 as i64;
                    self.projects[i] = None;
                    self.n_projects -= 1;
                }
            }
        } else if t==2{
            self.projects[m] = None;
            self.n_projects -= 1;
        } else if t==3{
            self.projects = vec![None; self.m];
            self.n_projects = 0;
        } else if t==4{
            self.l += 1;
        }
        self.hands[c] = None;
    }

    fn update_all_project(&mut self, projects: &Vec<(i64, usize)>){
        self.projects = projects.iter().map(|&c| Some(c)).collect_vec();
    }

    fn update_dummy_project(&mut self, projects: &Vec<(i64, usize)>){
        let mut idx = 0;
        for i in 0..self.projects.len(){
            if self.projects[i].is_none(){
                self.projects[i] = Some(projects[idx]);
                idx += 1;
            }
        }
        self.projects = projects.iter().map(|&c| Some(c)).collect_vec();
    }

    fn update_new_card(&mut self, card: (usize, usize, usize), swap_idx: usize){
        self.hands[swap_idx] = Some((card.0, card.1));
        self.money -= card.2 as i64;
        self.cur_turn += 1;
    }

    fn get_score(&self) -> f64{
        if self.money<0 || self.l>20{
            return f64::MIN;
        }
        let mut status_score = self.money as f64;
        // status_score += 100.0*2.0_f64.powf(self.l as f64);
        status_score += 1000.0;
        let mut n_cnt = 0;
        let mut l_cnt = 0;
        let mut hand_score = 0.0;
        let mut project_score = 0.0;
        // 雑スコア
        for cards in self.hands.iter(){
            if let Some((t, w)) = cards{
                if *t==0{
                    hand_score += *w as f64;
                } else if *t==1{
                    hand_score += (*w*self.m) as f64;
                } else if *t==2{
                    n_cnt += 1;
                } else if *t==3{
                    n_cnt += 4;
                } else if *t==4{
                    // hand_score += 100.0*2.0_f64.powf((self.l) as f64);
                    hand_score += 1000.0;
                    l_cnt += 1;
                }
            } else {
                hand_score += 1.0;
            }
        }
        for project in self.projects.iter(){
            if let Some((h, v)) = project{
                project_score += ((*v as f64 / (*h+1) as f64) + *v as f64 - *h as f64)/2.0;
            } else {
                project_score += 1.0;
            }
        }
        status_score + hand_score + project_score
    }
}

struct Game{
    state: State,
    n: usize,
    m: usize,
    k: usize,
    max_turn: usize,
    cur_turn: usize,
    l: usize,
}

// 乱数生成とかはここでやる。
impl Game{
    fn new(input: &Input) -> Self{
        let state = State::init_state(input);
        let n = input.n;
        let m = input.m;
        let k = input.k;
        let max_turn = input.t;
        Self{state, n, m, k, max_turn, cur_turn: 0, l:0}
    }

    fn decide_use_card(&mut self) -> (usize, usize){
        // TODO: 探索
        let mut best_param = (0, 0);
        let mut best_score = f64::MIN;
        for c in 0..self.n{
            for m in 0..self.m{
                let mut new_state = self.state.clone();
                new_state.update_use_card(c, m);
                let score = new_state.get_score();
                if score>best_score{
                    best_score = score;
                    best_param = (c, m);
                }
            }
        }
        best_param
    }

    fn decide_new_card(&self, cards: &Vec<(usize, usize, usize)>, swap_idx: usize) -> usize{
        let mut best_param = 0;
        let mut best_score = f64::MIN;
        for idx in 0..cards.len(){
            let mut new_state = self.state.clone();
            new_state.update_new_card(cards[idx], swap_idx);
            let score = new_state.get_score();
            if score>best_score{
                best_score = score;
                best_param = idx;
            }
        }
        best_param
    }

    fn process(&mut self){
        let (c, m) = self.decide_use_card();
        println!("{} {}", c, m);
        self.state.update_use_card(c, m);
        self.l = self.state.l;
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            cur_projects: [(i64, usize); self.m], // a is Vec<i32>, n-array.s
            money: i64,
            new_cards: [(usize, usize, usize); self.k], // a is Vec<i32>, n-array.
        }
        self.state.update_all_project(&cur_projects);
        let r = self.decide_new_card(&new_cards, c);
        println!("{}", r);
        self.state.update_new_card(new_cards[r], c);
        self.cur_turn += 1;
        eprintln!("turn: {}, money: {}", self.cur_turn, self.state.money);
    }

    fn run(&mut self){
        while self.cur_turn < self.max_turn{
            self.process();
        }
    }
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.8);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut game = Game::new(input);
    game.run();
}
