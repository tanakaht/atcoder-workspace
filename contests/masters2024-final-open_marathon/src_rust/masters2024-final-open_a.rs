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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Vector{
    x: i64,
    y: i64
}

impl Vector{
    fn norm(&self) -> f64{
        ((self.x*self.x+self.y*self.y) as f64).sqrt()
    }
}

impl std::ops::Mul<i64> for Vector {
    type Output = Vector;
    fn mul(self, rhs: i64) -> Self::Output {
        Vector{x: self.x*rhs, y: self.y*rhs}
    }
}

impl std::ops::Mul<f64> for Vector {
    type Output = Vector;
    fn mul(self, rhs: f64) -> Self::Output {
        Vector{x: (self.x as f64*rhs) as i64, y: (self.y as f64*rhs) as i64}
    }
}


impl std::ops::Add<Vector> for Vector {
    type Output = Vector;
    fn add(self, rhs: Vector) -> Self::Output {
        Vector{x: self.x+rhs.x, y: self.y+rhs.y}
    }
}

impl std::ops::Sub<Vector> for Vector {
    type Output = Vector;
    fn sub(self, rhs: Vector) -> Self::Output {
        Vector{x: self.x-rhs.x, y: self.y-rhs.y}
    }
}

#[derive(Debug, Clone, Copy)]
struct Line{
    l: Vector,
    r: Vector
}

impl Line{
    fn contact(&self, other: Line) -> bool{
        let p = self.r-self.l;
        let q1 = other.r-self.l;
        let q2 = other.l-self.l;
        if p.x==0{
            return q1.y*q2.y<=0;
        }
        let s = (q1.x*p.y)/p.x-q1.y;
        let t = (q2.x*p.y)/p.x-q2.y;
        return s*t<=0;
    }

    fn extend(&self, l: i64) -> Line{
        let p = self.r-self.l;
        Line{l: self.l-p*((l as f64/p.norm()) as i64), r: self.r+p*((l as f64/p.norm()) as i64)}
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    m: usize,
    eps: f64,
    sigma: f64,
    s: Vector,
    p: Vec<Vector>,
    l: Vec<Line>,
    points: Vec<Vector>,
    g: Vec<HashSet<usize>>,
}

impl Input {
    fn read_input() -> Self {
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            n: usize,
            m: usize,
            eps: f64,
            sigma: f64,
            s_: (i64, i64),
            //m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            p_: [(i64, i64); n], // `a` is Vec<Vec<i32>>, (m, n)-matrix.
            l_: [(i64, i64, i64, i64); m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let s = Vector{x: s_.0, y: s_.1};
        let mut p: Vec<Vector> = p_.iter().map(|&(x, y)| Vector{x, y}).collect();
        let mut l: Vec<Line> = l_.iter().map(|&(x1, y1, x2, y2)| Line{l: Vector{x:x1, y:y1}, r: Vector{x:x2, y:y2}}).collect();
        let mut points = vec![];
        for p_ in p.iter(){
            points.push(*p_);
        }
        for l_ in l.iter(){
            let l_extended = l_.extend(100);
            let mut flg = true;
            let l_extended_l = Line{l: l_extended.l, r: l_.l};
            for l__ in l.iter(){
                if l_extended_l.contact(*l__){
                    flg = false;
                    break;
                }
            }
            if flg{
                points.push(l_extended.l);
            }

            let mut flg = true;
            let l_extended_r = Line{l: l_.r, r: l_extended.r};
            for l__ in l.iter(){
                if l_extended_r.contact(*l__){
                    flg = false;
                    break;
                }
            }
            if flg{
                points.push(l_extended.r);
            }
        }
        points.push(s.clone());
        let mut g = vec![HashSet::new(); points.len()];
        for i in 0..points.len(){
            for j in i+1..points.len(){
                let  l_ = Line{l: points[i], r: points[j]};
                let mut flg = true;
                for l__ in l.iter(){
                    if l_.contact(*l__){
                        flg = false;
                        break;
                    }
                }
                if flg{
                    g[i].insert(j);
                    g[j].insert(i);
                }
            }
        }
        Self { n, m, eps, sigma, s, p, l, points, g }
    }
}

struct Drone{
    p: Vector,
    v: Vector
}

impl Drone{
    fn new(s: &Vector) -> Self{
        Self{p: s.clone(), v: Vector{x: 0, y: 0}}
    }

    fn moves(&mut self, to: Vector) -> Vec<Vector>{
        // TODO: bisect
        let mut turn = 1;
        loop{
            let norm = (to-self.p-self.v*turn).norm();
            if norm<=(500*turn*(turn+1)/2) as f64{
                break;
            }
            turn += 1;
        }
        let mut ans = vec![];
        let mut mokutekiti = to-self.p-self.v*turn;
        let norm = mokutekiti.norm();
        for t in 0..turn-1{
            let mut d = (mokutekiti.norm()+500.0*((turn-t-1)*(turn-t)/2) as f64)/(turn-t) as f64;
            if 500.0<=d{
                d=500.0;
            }
            let v = mokutekiti*(d/mokutekiti.norm());
            ans.push(v);
            mokutekiti = mokutekiti-v*(turn-t);
        }
        ans.push(mokutekiti);
        ans
    }

    fn process(&mut self, a: Vector, c: bool, q: Vec<usize>){
        // eprintln!("current pos: {} {}, move: {} {} v: {} {}", self.p.x, self.p.y, a.x, a.y, self.v.x, self.v.y);
        if c{
            self.v = Vector{x: 0, y: 0};
            return;
        }
        self.v = self.v+a;
        self.p = self.p+self.v;
    }
}

struct Neighbor{
    flg: usize,
    i: usize,
    j: usize,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    order: Vec<usize>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut order = vec![input.points.len()-1];
        let cur = input.points.len()-1;
        let mut used = vec![false; input.points.len()];
        used[cur] = true;
        // 適当に行ったことない点を選ぶ
        loop{
            let mut candidates = vec![];
            for i in input.g[cur].iter(){
                if !used[*i]{
                    candidates.push(*i);
                }
            }
            if candidates.len()==0{
                break;
            }
            let cur = candidates[0];
            used[cur] = true;
            order.push(cur);
        }
        Self {order}
    }

    fn update(&mut self, params: Neighbor)->Neighbor{
        if params.flg==0{
            // swap
            self.order.swap(params.i, params.j);
            params
        } else if params.flg==1{
            // 2-opt
            self.order[params.i..params.j].reverse();
            params
        } else if params.flg==2{
            // jをi番目に追加
            self.order.insert(params.i, params.j);
            Neighbor{flg: 3, i: params.i, j: 0}
        } else{
            // i番目の要素を削除
            let pre = self.order[params.i];
            self.order.remove(params.i);
            Neighbor{flg: 2, i: params.i, j: pre}
        }
    }

    fn undo(&mut self, params: Neighbor){
        self.update(params);
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let n = input.n;
        let mut rng = rand::thread_rng();
        let mode_flg = rng.gen::<usize>()%100;
        if mode_flg<40{
            // swap
            let mut i = rng.gen::<usize>()%(self.order.len()-2)+1;
            let mut j = rng.gen::<usize>()%(self.order.len()-2)+1;
            for _ in 0..1000{
                i = rng.gen::<usize>()%(self.order.len()-2)+1;
                j = rng.gen::<usize>()%(self.order.len()-2)+1;
                if i==j{
                    continue
                }
                if !input.g[self.order[i-1]].contains(&self.order[j]) || !input.g[self.order[j-1]].contains(&self.order[i]) || !input.g[self.order[i]].contains(&(self.order[j+1])) || !input.g[self.order[j]].contains(&(self.order[i+1])){
                    continue;
                }
                break;
            }
            Neighbor{flg: 0, i, j}
        } else if mode_flg<80{
            // 2-opt
            let mut i = rng.gen::<usize>()%(self.order.len()-2)+1;
            let mut j = rng.gen::<usize>()%(self.order.len()-2)+1;
            for _ in 0..1000{
                i = rng.gen::<usize>()%(self.order.len()-2)+1;
                j = rng.gen::<usize>()%(self.order.len()-2)+1;
                if i==j{
                    continue
                }
                if i>j{
                    std::mem::swap(&mut i, &mut j);
                }
                if !input.g[self.order[i-1]].contains(&self.order[j]) || !input.g[self.order[i]].contains(&(self.order[j+1])){
                    continue;
                }
                break;
            }
            Neighbor{flg: 1, i, j}
        } else if mode_flg<90{
            // jをi番目に追加
            let mut i = rng.gen::<usize>()%(self.order.len()-1)+1;
            let mut j = rng.gen::<usize>()%(input.points.len()-1);
            for _ in 0..1000{
                i = rng.gen::<usize>()%(self.order.len()-1)+1;
                j = rng.gen::<usize>()%(input.points.len()-1);
                if !input.g[self.order[i-1]].contains(&j) || !input.g[self.order[i]].contains(&(j+1)){
                    continue;
                }
                break;
            }
            Neighbor{flg: 2, i, j}
        } else {
            // i番目の要素を削除
            let mut i = rng.gen::<usize>()%(self.order.len()-1)+1;
            for _ in 0..1000{
                i = rng.gen::<usize>()%(self.order.len()-1)+1;
                if !input.g[self.order[i-1]].contains(&self.order[i]) || (i<self.order.len()-1 && !input.g[self.order[i+1]].contains(&self.order[i])){
                    continue;
                }
                break;
            }
            Neighbor{flg: 2, i, j:0}
        }
    }

    fn get_score(&self, input: &Input) -> i64{
        let mut drone = Drone::new(&input.s);
        let mut appeared = vec![false; input.points.len()];
        let mut turn = 0;
        for u in self.order.iter().skip(1){
            let moves = drone.moves(input.points[*u]);
            for a in moves.iter(){
                drone.process(*a, false, vec![]);
                turn += 1;
            }
            if turn >= 5000{
                break;
            }
            if *u<input.n{
                appeared[*u] = true;
            }
        }
        let mut score = turn*-2;
        for i in 0..input.points.len(){
            if appeared[i]{
                score += 1000;
            }
        }
        score
    }

    fn print(&mut self, input: &Input){
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        while self.order.len()>2 && self.order.last().unwrap()>=&input.n{
            self.order.pop();
        }
        let mut drone = Drone::new(&input.s);
        let mut turn = 0;
        for u in self.order.iter().skip(1){
            let mut accs = drone.moves(input.points[*u]);
            accs.reverse();
            loop{
                if accs.len()==0{
                    break;
                }
                if turn>=5000{
                    break;
                }
                let a = accs.pop().unwrap();
                println!("A {} {}", a.x, a.y);
                input! {
                    from &mut source,
                    c: usize,
                    h: usize,
                    q: [usize; h]
                }
                println!("#p_ {} {}", drone.p.x, drone.p.y);
                println!("#v_ {} {}", drone.v.x, drone.v.y);

                drone.process(a, c==1, q);
                turn += 1;
                if c==1{
                    accs = drone.moves(input.points[*u]);
                    accs.reverse();
                }
            }
        }
        while turn<5000{
            println!("A {} {}", 0, 0);
            input! {
                from &mut source,
                c: usize,
                h: usize,
                q: [usize; h]
            }
            turn += 1;
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
        let undo_neighbor = state.update(neighbor);
        let new_score = state.get_score(input);
        let score_diff = new_score-cur_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0 || rng.gen_bool((score_diff as f64/temp).exp()){
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
            state.undo(undo_neighbor);
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
    eprintln!("1");
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
}
