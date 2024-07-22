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

// TODO: そんな変わらん気がするけど高速化?
fn inversion(xs: &Vec<usize>) -> usize {
    let mut ret = 0;
    for (i, v) in xs.iter().enumerate() {
        for v_ in xs[i + 1..].iter() {
            if v < v_ {
                ret += 1;
            }
        }
    }
    ret
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    m: usize,
    b: Vec<Vec<usize>>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            b_: [[usize; n/m]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let mut b = vec![];
        b.push(vec![]);
        for x in b_.iter(){
            b.push(x.clone());
        }
        Self { n, m, b }
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct State{
    b: Vec<Vec<usize>>,
    op: Vec<(usize, usize)>,
    score: Option<i32>,
    hp: usize,
    place: Vec<(usize, usize)>,
    serial: String,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let b = input.b.clone();
        let mut place = vec![(0, 0); input.n+1];
        for (i, x) in b.iter().enumerate(){
            for (j, y) in x.iter().enumerate(){
                place[*y] = (i, j);
            }
        }
        let mut ret = Self {b, op: vec![], score: None, hp: 0, place, serial: "".to_string()};
        ret.score = Some(ret.get_score());
        ret
    }

    fn update(&mut self, params: &(usize, usize)){
        let (to, v) = params;
        let (fr, idx) = self.place[*v];
        self.op.push((*to, *v));
        if *to==0{
            assert!(self.b[fr].len()==idx+1);
            self.b[0].push(*v);
            self.b[fr].pop();
            self.serial += &format!("{} {} ", to, v);
            return;
        }
        self.score = Some(self.score.unwrap()+self.get_score_diff(params));
        // 移動
        let eles = self.b[fr][idx..].to_vec();
        for v_ in eles.iter(){
            self.b[*to].push(*v_);
            self.place[*v_] = (*to, self.b[*to].len()-1);
        }
        // 移動元の削除
        self.b[fr].truncate(idx);
        // serialの更新
        self.serial += &format!("{} {} ", to, v);
        // 取れる限りとっちゃう
        loop {
            let cur_idx = (*self.b[0].last().unwrap_or(&0))+1;
            if cur_idx==201{
                break;
            }
            if self.place[cur_idx].1!=self.b[self.place[cur_idx].0].len()-1{
                break;
            }
            self.b[0].push(cur_idx);
            self.b[self.place[cur_idx].0].pop();
            self.op.push((0, cur_idx));
            self.serial += &format!("{} {} ", 0, cur_idx);
        }
        // TODO: scoreの差分計算
        self.hp += eles.len()+1;
    }

    fn get_neighbors(&self) -> Vec<(usize, usize)>{
        let mut rng = rand::thread_rng();
        let cur_idx = (*self.b[0].last().unwrap_or(&0))+1;
        if cur_idx==201{
            return vec![];
        }
        let mut ret = vec![];
        let (fr, idx) = self.place[cur_idx];
        if idx==self.b[fr].len()-1{
            return vec![(0, cur_idx)];
        }
        // TODO: ある程度枝刈り?
        for idx_ in idx+1..(self.b[fr].len()){
        // for idx_ in idx+1..(idx+2){
            if self.b[fr][idx_]<self.b[fr][idx_-1]{
                continue;
            }
            for to in 1..11{
                if fr==to{
                    continue;
                }
                ret.push((to, self.b[fr][idx_]));
            }
        }
        // 直近のいくつかをどっかに逃しておく
        // for i in cur_idx+1..cur_idx+2{
        //     if i>=201{
        //         break;
        //     }
        //     let (fr, idx) = self.place[i];
        //     for to in 1..11{
        //         if fr==to{
        //             continue;
        //         }
        //         ret.push((to, i));
        //     }
        // }

        ret
    }


    fn get_score_diff(&self, params: &(usize, usize)) -> i32{
        if self.op.len()>=5000{
            return -100000;
        }
        let (to, v) = params;
        let (fr, idx) = self.place[*v];
        let mut inv_cnts_diff = 0;
        inv_cnts_diff -= inversion(&self.b[fr]) as i32;
        inv_cnts_diff -= inversion(&self.b[*to]) as i32;
        inv_cnts_diff += inversion(&self.b[fr][..idx].to_vec()) as i32;
        // self.b[*to]にself.b[fr][idx..]をappendしたものをtmpとする
        let mut tmp = self.b[*to].clone();
        tmp.append(&mut self.b[fr][idx..].to_vec());
        inv_cnts_diff += inversion(&tmp) as i32;
        let hp_diff = (self.b[fr].len()-idx+1) as i32;
        let mut score_diff = 0;
        score_diff += hp_diff*2;
        score_diff += inv_cnts_diff*4;
        score_diff
    }


    fn get_score(&self) -> i32{
        if self.op.len()>5000{
            return -100000;
        }
        let mut inv_cnts = 0;
        for b in self.b[1..].iter(){
            inv_cnts += inversion(b);
        }
        let mut score = 0;
        score += self.hp as i32*2;
        score += (inv_cnts as i32)*4;
        score
    }

    fn print(&mut self){
        for op in self.op.iter(){
            println!("{} {}", op.1, op.0);
        }
    }
}


// TODO: chatgpt謹製
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // 比較の優先順位に従って、selfとotherを比較
        match (self.score, other.score, self.serial.cmp(&other.serial)) {
            (Some(self_score), Some(other_score), _) => {
                // scoreが両方ともSomeの場合
                other_score.cmp(&self_score).then(self.serial.cmp(&other.serial))
            }
            (Some(_), None, _) => Ordering::Less, // selfのscoreがSomeで、otherのscoreがNoneの場合はselfが優先
            (None, Some(_), _) => Ordering::Greater, // selfのscoreがNoneで、otherのscoreがSomeの場合はotherが優先
            (None, None, _) => self.serial.cmp(&other.serial), // どちらもscoreがNoneの場合はserialで比較
        }
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn beam_search(init_state: &State, beam_width: usize, timer: &Instant, tl: f64) -> State {
    let mut beam: BinaryHeap<State> = BinaryHeap::new();
    let mut best_state: State = init_state.clone();
    beam.push(init_state.clone());
    let mut cnt = 0;
    while !beam.is_empty() {
        if timer.elapsed().as_secs_f64() > tl {
            break;
        }
        let mut next_beam_kouho: BinaryHeap<(Reverse<i32>, (usize, usize), usize)> = BinaryHeap::new();
        let mut beam_seeds = vec![];
        for seed_idx in 0..beam_width {
            if timer.elapsed().as_secs_f64() > tl {
                break;
            }
            if let Some(state) = beam.pop() {
                if state.b[0].len() == 200 {
                    if best_state.b[0].len()!=200 || best_state.score > state.score {
                        best_state = state.clone();
                    }
                } else {
                    for (to, v) in state.get_neighbors() {
                        cnt += 1;
                        let score = state.score.unwrap() + state.get_score_diff(&(to, v));
                        next_beam_kouho.push((Reverse(score), (to, v), seed_idx));
                        if timer.elapsed().as_secs_f64() > tl {
                            break;
                        }

                    }
                }
                beam_seeds.push(state);
            }
        }
        beam = BinaryHeap::new();
        for _ in 0..beam_width {
            if timer.elapsed().as_secs_f64() > tl {
                break;
            }
            if let Some((_, params, seed_idx)) = next_beam_kouho.pop() {
                let mut new_state = beam_seeds[seed_idx].clone();
                new_state.update(&params);
                beam.push(new_state);
            }
        }

    }
    eprintln!("cnt: {}", cnt);
    best_state
}



fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.9);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut init_state = State::init_state(input);
    let mut best_state1 = beam_search(&init_state, 1, timer, tl);
    let mut best_state2 = beam_search(&init_state, 200, timer, tl);
    if best_state2.b[0].len() == 200 {
        best_state2.print();
    } else {
        // assert!(best_state2.b[0].len() == 200);
        best_state1.print();
    }
}
