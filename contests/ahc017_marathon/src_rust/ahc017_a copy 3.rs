#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::HashSet;
use itertools::Itertools;
use num::range;
use proconio::*;
use std::iter::FromIterator;
use std::collections::BinaryHeap;
use std::fmt::Display;
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;

const INF: usize = 1 << 31;
const SEED: u128 = 42;

#[allow(unused_macros)]
macro_rules! chmin {
    ($base:expr, $($cmps:expr),+ $(,)*) => {{
        let cmp_min = min!($($cmps),+);
        if $base > cmp_min {
            $base = cmp_min;
            true
        } else {
            false
        }
    }};
}

#[allow(unused_macros)]
macro_rules! chmax {
    ($base:expr, $($cmps:expr),+ $(,)*) => {{
        let cmp_max = max!($($cmps),+);
        if $base < cmp_max {
            $base = cmp_max;
            true
        } else {
            false
        }
    }};
}

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

#[allow(unused_macros)]
macro_rules! mat {
    ($e:expr; $d:expr) => { vec![$e; $d] };
    ($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

#[derive(Debug, Clone, Copy)]
struct Point{
    x: usize,
    y: usize
}
impl Point{
    fn mean(&self, p: &Point) -> Point{
        Point{x: (self.x+p.y)/2, y:(self.y+p.y)/2}
    }
    fn dist(&self, p: &Point) -> usize{
        (self.x+p.y).pow(2)+(self.y+p.y).pow(2)
    }
}

struct Dijkstra {
    dists: Vec<usize>,
    start_node: usize,
    end_node: Option<usize>,
}

impl Dijkstra {
    // n:usize 頂点の数
    // edge: Vec<Vec<(usize,usize)>> edge[i] = [(2,3), (3,1), (頂点への道,重み)]
    // init:usize どの頂点を起点に考えるか

    fn new(G: &Vec<Vec<(usize, usize, usize)>>, start_node: usize, end_node: Option<usize>, removed_edge: &HashSet<usize>) -> Self {
        let N = G.len();
        let mut q: BinaryHeap<(usize, usize)> = BinaryHeap::new();
        let mut dists = vec![INF; N];
        let mut appeared = vec![false; N];
        dists[start_node] = 0;
        q.push((INF, start_node));
        while let Some((d_, u)) = q.pop() {
            let d = INF-d_;
            if Some(u)==end_node {
                break;
            }
            if appeared[u]{
                continue;
            }
            appeared[u] = true;
            for (v, c, i) in G[u].iter(){
                if removed_edge.contains(i){
                    continue;
                }
                let d2 = d+c;
                if dists[*v] > d2 {
                    dists[*v] = d2;
                    q.push((INF-d2, *v))
                }
            }
        }
        Self { dists, start_node, end_node }
    }

    fn distance(&self, target: usize) -> usize {
        self.dists[target]
    }

    fn total_distance(&self) -> usize {
        self.dists.iter().sum()
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    N: usize,
    M: usize,
    D: usize,
    K: usize,
    UVW: Vec<(usize, usize, usize)>,
    XY: Vec<Point>,
    G: Vec<Vec<(usize, usize, usize)>>,
    depends_pair: HashSet<(usize, usize)>,
    edge_center: Vec<Point>
}

impl Input {
    fn read_input() -> Self {
        input! {
            N: usize,
            M: usize,
            D: usize,
            K: usize,
            UVW_: [[usize; 3]; M],
            XY_: [[usize; 2]; N]
        }
        let UVW: Vec<(usize, usize, usize)> = UVW_.iter().map(|x| (x[0]-1, x[1]-1, x[2])).collect();
        let XY: Vec<Point> = XY_.iter().map(|ele| Point{x: ele[0], y:ele[1]}).collect();
        let mut G: Vec<Vec<(usize, usize, usize)>> = vec![vec![]; N];
        let mut depends_pair: HashSet<(usize, usize)> = HashSet::new();
        let mut edge_center: Vec<Point> = vec![];
        for (i, (u, v, w)) in UVW.iter().enumerate(){
            G[*u].push((*v, *w, i));
            G[*v].push((*u, *w, i));
        }
        for (i, (start_node, end_node, w)) in UVW.iter().enumerate(){
            let dijkstra = Dijkstra::new(&G, *start_node, Some(*end_node), &HashSet::from_iter(vec![i].iter().cloned()));
            let mut cur = *end_node;
            // 最短経路復元してdepends_pairに入れる
            while cur!=*start_node{
                for (v, c, j) in G[cur].iter(){
                    if dijkstra.distance(*v)+c==dijkstra.distance(cur){
                        cur = *v;
                        depends_pair.insert((i, *j));
                        depends_pair.insert((*j, i));
                        break;
                    }
                }
            }
        }
        for (i, (u, v, w)) in UVW.iter().enumerate(){
            edge_center.push(XY[*u].mean(&XY[*v]));
        }
        Self { N, M, D, K, UVW, XY, G, depends_pair, edge_center }
    }
}


#[derive(Debug, Clone)]
struct State {
    assigns: Vec<HashSet<usize>>,
    ans: Vec<usize>,
    sample_nodes: HashSet<usize>,
    daily_score: Vec<usize>,
}

impl State {
    fn init(input: &Input, seed: u128) -> Self {
        let mut rng = rand::thread_rng();
        let mut assigns: Vec<HashSet<usize>> = vec![HashSet::new(); input.D];
        // 初期アサイン
        let mut UVW_enumerate: Vec<(usize, &(usize, usize, usize))> = input.UVW.iter().enumerate().into_iter().collect();
        let mut best_total_depend_cnt = INF;
        for _ in 0..10{
            let mut assigns_tmp: Vec<HashSet<usize>> = vec![HashSet::new(); input.D];
            UVW_enumerate.shuffle(&mut rng);
            let mut total_depend_cnt = 0;
            for (i, (u, v, w)) in UVW_enumerate.iter(){
                let mut best_d: usize = 0;
                let mut best_metrics = (INF, INF);
                for d in 0..input.D{
                    if assigns_tmp[d].len()==input.K{
                        continue;
                    }
                    let mut depend_cnt: usize = 0;
                    for x in assigns_tmp[d].iter(){
                        if input.depends_pair.contains(&(*x, *i)){
                            depend_cnt += 1
                        }
                    }
                    // let depend_cnt: usize = assigns_tmp[d].iter().map(|x| input.depends_pair.contains(&(*i, *x)) as usize).sum();
                    let metrics = (depend_cnt, assigns_tmp[d].len());
                    if metrics<best_metrics{
                        best_metrics = metrics;
                        best_d = d;
                    }
                }
                assigns_tmp[best_d].insert(*i);
                total_depend_cnt += best_metrics.0;
            }
            if total_depend_cnt<best_total_depend_cnt{
                best_total_depend_cnt = total_depend_cnt;
                assigns = assigns_tmp;
            }
            if best_total_depend_cnt==0{
                break;
            }
        }

        // ansを反映
        let mut ans = vec![0; input.M];
        for (d, removed_edge) in assigns.iter().enumerate(){
            for e in removed_edge.iter(){
                ans[*e] = d;
            }
        }
        // sample_nodes
        let v:Vec<usize> = (0..input.M).collect();
        let sample_nodes: HashSet<usize> = HashSet::from_iter(v.choose_multiple(&mut rng, 20).cloned().collect_vec());
        // daily_score
        let daily_score: Vec<usize> = (0..input.D).map(|d| State::get_daily_score_with_sample_nodes(input, &assigns[d], &sample_nodes)).collect();
        Self { assigns, ans, sample_nodes, daily_score }
    }

    fn get_neighbor(&self, input: &Input) -> (&str, (usize, usize, usize, usize)){
        let mut rng = rand::thread_rng();
        if rng.gen::<usize>()%10 < 8{
            for _ in 0..100{
                let d1 = rng.gen::<usize>()%input.D;
                let d2 = rng.gen::<usize>()%input.D;
                if (d1==d2 || self.assigns[d1].is_empty() || self.assigns[d2].is_empty()){
                    continue;
                }
                let u1 = *self.assigns[d1].iter().next().unwrap();
                let u2 = *self.assigns[d2].iter().next().unwrap();
                return ("swap", (d1, u1, d2, u2));
            }
        } else {
            for _ in 0..100{
                let d1 = rng.gen::<usize>()%input.D;
                let d2 = rng.gen::<usize>()%input.D;
                if (d1==d2 || self.assigns[d1].is_empty() || self.assigns[d2].len()==input.K){
                    continue;
                }
                let u1 = *self.assigns[d1].iter().next().unwrap();
                let u2 = INF;
                return ("move", (d1, u1, d2, u2));
            }
        }
        ("none", (INF, INF, INF, INF))
    }

    fn update(&mut self, mode: &str, params: (usize, usize, usize, usize)){
        if mode=="swap"{
            let (d1, u1, d2, u2) = params;
            self.assigns[d1].remove(&u1);
            self.assigns[d2].remove(&u2);
            self.assigns[d1].insert(u2);
            self.assigns[d2].insert(u1);
            self.ans[u1] = d2;
            self.ans[u2] = d1;
        } else if mode=="move"{
            let (d1, u1, d2, u2) = params;
            self.assigns[d1].remove(&u1);
            self.assigns[d2].insert(u1);
            self.ans[u1] = d2;
        }
    }

    fn undo(&mut self, mode: &str, params: (usize, usize, usize, usize)){
        if mode=="swap"{
            let (d1, u1, d2, u2) = params;
            self.update(mode, (d1, u2, d2, u1));
        } else if mode=="move"{
            let (d1, u1, d2, u2) = params;
            self.update(mode, (d2, u1, d1, u2));
        }
    }

    fn get_daily_score_with_sample_nodes(input: &Input, removed_edges: &HashSet<usize>, sample_nodes: &HashSet<usize>) -> usize{
        let mut score: usize = 0;
        for u in sample_nodes.iter(){
            let dijkstra = Dijkstra::new(&input.G, *u, None, removed_edges);
            let sabun: usize = dijkstra.dists.iter().sum();
            score += sabun;
        }
        score
    }

    fn get_daily_score(input: &Input, removed_edges: &HashSet<usize>) -> usize{
        let mut score: usize = 0;
        for u in 0..input.N{
            let dijkstra = Dijkstra::new(&input.G, u, None, removed_edges);
            let sabun: usize = dijkstra.dists.iter().sum();
            score += sabun;
        }
        score
    }

    fn get_score(&self, input: &Input) -> usize {
        let mut score = 0;
        for d in 0..input.D{
            score += State::get_daily_score(input, &self.assigns[d]);
        }
        score
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dst: Vec<String> = self.ans.iter().map(|x| (x+1).to_string()).collect();
        write!(f, "{}", dst.join(" "))?;
        Ok(())
    }
}

fn climbing(input: &Input, init_state: State, timer: &Instant, limit: f64){
    let mut state = init_state;
    while timer.elapsed().as_secs_f64()<limit{
        let (mode, params) = state.get_neighbor(input);
        let (d1, u1, d2, u2) = params;
        let pre_score_d1 = state.daily_score[d1];
        let pre_score_d2 = state.daily_score[d2];
        let aft_score_d1: usize;
        let aft_score_d2: usize;
        if mode=="swap"{
            state.update(mode, params);
            aft_score_d1 = State::get_daily_score_with_sample_nodes(input, &state.assigns[d1], &state.sample_nodes);
            aft_score_d2 = State::get_daily_score_with_sample_nodes(input, &state.assigns[d2], &state.sample_nodes);
            state.undo(mode, params);
        } else {
            aft_score_d1 = 0;
            aft_score_d2 = 0;
        }
        let score_diff = (aft_score_d1+aft_score_d2)-(pre_score_d1+pre_score_d2);
        if score_diff<=0{
            state.update(mode, params);
            state.daily_score[d1] = aft_score_d1;
            state.daily_score[d2] = aft_score_d2;
        }
    }
}


fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    let state = solve(&input);
    for _ in 0..2{
        state.get_score(&input);
    }
    println!("{}", &state);
}


fn solve(input: &Input) -> State {
    let init_state = State::init(input, SEED);
    init_state
}
