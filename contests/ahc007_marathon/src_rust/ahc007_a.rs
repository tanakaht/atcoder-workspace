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
use proconio::source::line::LineSource;
use std::io::BufReader;
use proconio::input_interactive;

const N: usize = 400;
const M: usize = 1995;
const N_ITER: usize = 220;

#[derive(Debug, Clone)]
struct UnionFind {
    n: usize,
    parent_or_size: Vec<i32>,
}

impl UnionFind {
    pub fn new(size: usize) -> Self {
        Self {
            n: size,
            parent_or_size: vec![-1; size],
        }
    }

    pub fn union(&mut self, a: usize, b: usize) -> usize {
        assert!(a < self.n);
        assert!(b < self.n);
        let (mut x, mut y) = (self.find(a), self.find(b));
        if x == y {
            return x;
        }
        if -self.parent_or_size[x] < -self.parent_or_size[y] {
            std::mem::swap(&mut x, &mut y);
        }
        self.parent_or_size[x] += self.parent_or_size[y];
        self.parent_or_size[y] = x as i32;
        x
    }

    pub fn same(&mut self, a: usize, b: usize) -> bool {
        assert!(a < self.n);
        assert!(b < self.n);
        self.find(a) == self.find(b)
    }

    pub fn find(&mut self, a: usize) -> usize {
        assert!(a < self.n);
        if self.parent_or_size[a] < 0 {
            return a;
        }
        self.parent_or_size[a] = self.find(self.parent_or_size[a] as usize) as i32;
        self.parent_or_size[a] as usize
    }

    pub fn size(&mut self, a: usize) -> usize {
        assert!(a < self.n);
        let x = self.find(a);
        -self.parent_or_size[x] as usize
    }

    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        let mut find_buf = vec![0; self.n];
        let mut group_size = vec![0; self.n];
        for i in 0..self.n {
            find_buf[i] = self.find(i);
            group_size[find_buf[i]] += 1;
        }
        let mut result = vec![Vec::new(); self.n];
        for i in 0..self.n {
            result[i].reserve(group_size[i]);
        }
        for i in 0..self.n {
            result[find_buf[i]].push(i);
        }
        result
            .into_iter()
            .filter(|x| !x.is_empty())
            .collect::<Vec<Vec<usize>>>()
    }
}

fn sample_range(low: usize, high: usize, rng: &mut ThreadRng) -> usize {
    low + rng.gen::<usize>()%(high-low+1)
}

#[allow(unused_variables)]
#[derive(Debug, Clone, Copy)]
struct Point {
    x: usize,
    y: usize,
}

impl Point {
    fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    fn dist(&self, p: &Point) -> usize {
        let dx = self.x.abs_diff(p.x);
        let dy = self.y.abs_diff(p.y);
        ((dx * dx + dy * dy) as f64).sqrt().round() as usize
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    points: Vec<Point>,
    edges: Vec<(usize, usize, usize)>,
    graph: Vec<Vec<(usize, usize)>>,
    turn: usize,
    uf: UnionFind,
    sample_edges: Vec<Vec<(usize, usize, usize, usize)>>,
}

impl Input {
    fn read_input() -> Self {
        // let stdin = std::io::stdin();
        // let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input_interactive! {
            // from &mut source,
            xy: [[usize; 2]; N],
            ab: [[usize; 2]; M],
            //m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let points: Vec<Point> = xy.iter().map(|p| Point::new(p[0], p[1])).collect();
        let edges: Vec<(usize, usize, usize)> = ab.iter().map(|e| (points[e[0]].dist(&points[e[1]]), e[0], e[1])).collect();
        let mut graph = vec![vec![]; N];
        for (l, a, b) in edges.iter(){
            graph[*a].push((*l, *b));
            graph[*b].push((*l, *a));
        }
        let uf = UnionFind::new(N);
        let mut sample_edges = vec![];
        let mut rng = thread_rng();
        for _ in 0..N_ITER{
            let mut tmp_edges = vec![];
            for (i, (l, a, b)) in edges.iter().enumerate(){
                tmp_edges.push((sample_range(*l, 3*l, &mut rng), *a, *b, i));
            }
            tmp_edges.sort_by_key(|x| x.0);
            sample_edges.push(tmp_edges);
        }
        Self { points, edges, graph, turn: 0, uf, sample_edges }
    }

    fn gen_sample_single_edge(&self, i: usize, rng: &mut ThreadRng) -> usize {
        let l = self.edges[i].0;
        sample_range(l, 3*l, rng)
    }

    fn gen_sample_edges(&self, rng: &mut ThreadRng) -> Vec<(usize, usize, usize)> {
        let mut edges = vec![];
        for (l, a, b) in self.edges[self.turn..].iter(){
            edges.push((sample_range(*l, 3*l, rng), *a, *b));
        }
        edges
    }

    fn get_actual_len(&mut self) -> usize{
        // let stdin = std::io::stdin();
        // let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input_interactive! {
            // from &mut source,
            l_act: usize,
        }
        let (l, a, b) = self.edges[self.turn];
        self.graph[a].retain(|x| *x!=(l, b));
        self.graph[b].retain(|x| *x!=(l, a));
        self.turn += 1;
        l_act
    }

    fn min_dist(&mut self, a: usize, b: usize) -> usize{
        let mut dists = vec![usize::MAX; N];
        let mut goals: HashSet<usize> = HashSet::new();
        let mut que = BinaryHeap::new();
        for xs in self.uf.groups().iter(){
            if self.uf.same(a, xs[0]){
                for x in xs.iter(){
                    dists[*x] = 0;
                    que.push((Reverse(0), *x));
                }
            } else if self.uf.same(b, xs[0]){
                goals = xs.iter().map(|&y| y).collect();
            }
        }
        while let Some((Reverse(d), u)) = que.pop(){
            if goals.contains(&u){
                return d;
            }
            if dists[u]<d{
                continue;
            }
            for &(l, v) in self.graph[u].iter(){
                let d_ = if (self.uf.same(u, v)){d} else {d+l};
                if dists[v]>d_{
                    dists[v] = d_;
                    que.push((Reverse(d_), v));
                }
            }
        }
        dists[b]
    }

    fn accept_edge(&mut self){
        let (l, a, b) = self.edges[self.turn-1];
        self.uf.union(a, b);
        self.graph[a].push((l, b));
        self.graph[b].push((l, a));
    }

    fn reject_edge(&mut self){
    }
}

fn main() {
    let timer = Instant::now();
    let mut input = Input::read_input();
    solve(&mut input, &timer, 1.8);
}

fn should_accept(input: &Input, l_act: usize, timer:&Instant, tl: f64) -> bool{
    let mut total_cnt = 0;
    let mut reject_cnt = 0;
    let (_, target_a, target_b) = input.edges[input.turn-1];
    let mut rng = thread_rng();
    for edges in input.sample_edges.iter(){
        if total_cnt>=4 && timer.elapsed().as_secs_f64()>tl{
            break;
        }
        let mut uf = input.uf.clone();
        total_cnt += 1;
        for &(l, a, b, i) in edges{
            if i<input.turn{
                continue;
            }
            if l>=l_act{
                break;
            }
            uf.union(a, b);
            if uf.same(target_a, target_b){
                reject_cnt += 1;
                break;
            }
        }
    }
    reject_cnt <= total_cnt/2
}


fn get_kitaiti_diff(input: &Input, timer:&Instant, tl: f64) -> usize{
    let mut e1 = 0;
    let mut e2 = 0;
    let (_, target_a, target_b) = input.edges[input.turn-1];
    let mut rng = thread_rng();
    for edges in input.sample_edges.iter(){
        if timer.elapsed().as_secs_f64()>tl{
            return usize::MAX;
        }
        let mut uf1 = input.uf.clone();
        let mut uf2 = input.uf.clone();
        uf1.union(target_a, target_b);
        let mut e1_ = 0;
        let mut e2_ = 0;
        for &(l, a, b, i) in edges{
            if i<input.turn{
                continue;
            }
            if !uf1.same(a, b){
                e1_ += l;
                uf1.union(a, b);
            }
            if !uf2.same(a, b){
                e2_ += l;
                uf2.union(a, b);
            }
            if uf1.size(0)==N && uf2.size(0)==N{
                break;
            }
        }
        if uf2.size(0)<N{
            return usize::MAX;
        }
        e1 += e1_;
        e2 += e2_;
    }
    (e2 - e1)/N_ITER
}

fn solve(input: &mut Input, timer:&Instant, tl: f64){
    for t in 0..M{
        let (l, a, b) = input.edges[input.turn];
        let l_act = input.get_actual_len();
        if input.uf.same(a, b){
            println!("0");
            input.reject_edge();
            continue;
        }
        let min_dist = input.min_dist(a, b);
        if 2*min_dist<l_act{
            println!("0");
            input.reject_edge();
            continue;
        } else if min_dist==usize::MAX{
            println!("1");
            input.accept_edge();
            continue;
        }
        // if get_kitaiti_diff(input,  timer, tl)>l_act{
        if should_accept(input, (l_act as f64*0.95) as usize, timer, tl){
            println!("1");
            input.accept_edge();
        } else {
            println!("0");
            input.reject_edge();
        }
    }
    eprintln!("time: {}ms", timer.elapsed().as_millis());
}
