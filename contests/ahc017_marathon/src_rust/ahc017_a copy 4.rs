#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::HashSet;
use itertools::Itertools;
use num::{range, abs};
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

#[derive(Debug, Clone)]
struct Dijkstra {
    dists: Vec<usize>,
    start_node: usize,
    end_node: Option<usize>,
    parents: Vec<usize>,
    used_edges: HashSet<usize>,
    total_dist: usize,
}

impl Dijkstra {
    fn new(G: &Vec<Vec<(usize, usize, usize)>>, start_node: usize, end_node: Option<usize>, removed_edge: &HashSet<usize>) -> Self {
        let N = G.len();
        let mut q: BinaryHeap<(usize, usize, usize, usize)> = BinaryHeap::new();
        let mut dists = vec![1000000000; N];
        let mut appeared = vec![false; N];
        let mut parents = vec![0; N];
        // TODO: vecにして後で収集
        let mut used_edges: HashSet<usize> = HashSet::new();
        dists[start_node] = 0;
        q.push((INF, start_node, start_node, INF));
        while let Some((d_, u, fr_u, fr_i)) = q.pop() {
            let d = INF-d_;
            if Some(u)==end_node {
                break;
            }
            if appeared[u]{
                continue;
            }
            appeared[u] = true;
            parents[u] = fr_u;
            used_edges.insert(fr_i);
            for (v, c, i) in G[u].iter(){
                let d2 = d+c;
                if(dists[*v]<=d2 || removed_edge.contains(i)){
                    continue;
                }
                dists[*v] = d2;
                q.push((INF-d2, *v, u, *i))
            }
        }
        let total_dist = dists.iter().sum();
        Self { dists, start_node, end_node, parents, used_edges, total_dist }
    }

    fn distance(&self, target: usize) -> usize {
        self.dists[target]
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
    daily_dijkstra: Vec<Vec<Dijkstra>>,
    daily_unused_edges: Vec<HashSet<usize>>,
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
                    let mut ucnt: usize = 0;
                    let mut vcnt: usize = 0;
                    for j in assigns_tmp[d].iter(){
                        if input.depends_pair.contains(&(*j, *i)){
                            depend_cnt += 1
                        }
                        let (u_, v_, w_) = input.UVW[*j];
                        if (*u==u_ || *u==w_){
                            ucnt += 1
                        }
                        if (*v==u_ || *v==w_){
                            vcnt += 1
                        }
                    }
                    let path_likelity = 100-100*((ucnt==1 && vcnt==1) as usize)-50*((ucnt+vcnt==1) as usize)+(ucnt+vcnt);
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
        // center, ru, rd, ld, lu
        // let mut sample_nodes_ = vec![0; 5];
        // let mut XY_enumerate: Vec<(usize, &Point)> = input.XY.iter().enumerate().into_iter().collect();
        // let center = Point{ x: 500, y:500};
        // XY_enumerate.shuffle(&mut rng);
        // for (u, p) in input.XY.iter().enumerate(){
        //     if p.dist(&center)<40000 {
        //         sample_nodes_[0] = u;
        //     } else if (p.x>=500 && p.y<500){
        //         sample_nodes_[1] = u;
        //     } else if (p.x>=500 && p.y>=500){
        //         sample_nodes_[2] = u;
        //     } else if (p.x<500 && p.y>=500){
        //         sample_nodes_[3] = u;
        //     } else if (p.x<500 && p.y<500){
        //         sample_nodes_[4] = u;
        //     }
        // }
        // let sample_nodes = HashSet::from_iter(sample_nodes_);
        let v:Vec<usize> = (0..input.N).collect();
        let sample_nodes: HashSet<usize> = HashSet::from_iter(v.choose_multiple(&mut rng, 5).cloned().collect_vec());
        // daily_score等
        let daily_dijkstra = assigns.iter().map(|assign| State::get_daily_dijkstra(input, assign, &sample_nodes)).collect_vec();
        let daily_score: Vec<usize> = daily_dijkstra.iter().map(|dijkstras| State::get_daily_score_from_dijkstras(dijkstras)).collect();
        let mut daily_unused_edges = vec![];
        for (dijkstras, assign) in daily_dijkstra.iter().zip(assigns.iter()){
            daily_unused_edges.push(State::get_unused_edges(input, assign, dijkstras));
        }
        Self { assigns, ans, sample_nodes, daily_score, daily_dijkstra, daily_unused_edges }
    }

    fn get_neighbor(&self, input: &Input) -> (String, (usize, usize, usize, usize)){
        let mut rng = rand::thread_rng();
        let flg = rng.gen::<usize>()%100;
        for d2 in 0..input.D{
            if self.assigns[d2].len()==input.K{
                continue;
            }
            for u1 in self.daily_unused_edges[d2].iter(){
                let d1 = self.ans[*u1];
                let (u, v, w) = input.UVW[*u1];
                let mut is_shorter = false;
                for dijkstra in self.daily_dijkstra[d1].iter(){
                    let du = dijkstra.distance(u);
                    let dv = dijkstra.distance(v);
                    if (du+w<dv || dv+w<du){
                        is_shorter = true;
                        break;
                    }
                }
                if !is_shorter{
                    continue;
                }
                let u2 = INF;
                eprintln!("1");
                return ("move".to_string(), (d1, *u1, d2, u2));
            }
        }
        if flg < 80{
            // 依存無視swap
            for _ in 0..100{
                let d1 = rng.gen::<usize>()%input.D;
                let d2 = rng.gen::<usize>()%input.D;
                if (d1==d2 || self.assigns[d1].is_empty() || self.assigns[d2].is_empty()){
                    continue;
                }
                let u1 = *self.assigns[d1].iter().choose(&mut rng).unwrap();
                let u2 = *self.assigns[d2].iter().choose(&mut rng).unwrap();
                return ("swap".to_string(), (d1, u1, d2, u2));
            }
        } else if flg < 100 {
            // 依存無視move
            for _ in 0..100{
                let d1 = rng.gen::<usize>()%input.D;
                let d2 = rng.gen::<usize>()%input.D;
                if (d1==d2 || self.assigns[d1].is_empty() || self.assigns[d2].len()==input.K){
                    continue;
                }
                let u1 = *self.assigns[d1].iter().choose(&mut rng).unwrap();
                let u2 = INF;
                return ("move".to_string(), (d1, u1, d2, u2));
            }
        }
        ("none".to_string(), (INF, INF, INF, INF))
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

    fn update_with_score(&mut self, input: &Input, mode: &str, params: (usize, usize, usize, usize), daily_dijkstra: (Vec<Dijkstra>, Vec<Dijkstra>)){
        self.update(mode, params);
        let (d1, u1, d2, u2) = params;
        let (dijkstra_d1, dijkstra_d2) = daily_dijkstra;
        self.daily_score[d1] = State::get_daily_score_from_dijkstras(&dijkstra_d1);
        self.daily_unused_edges[d1] = State::get_unused_edges(input, &self.assigns[d1], &dijkstra_d1);
        self.daily_dijkstra[d1] = dijkstra_d1;
        self.daily_score[d2] = State::get_daily_score_from_dijkstras(&dijkstra_d2);
        self.daily_unused_edges[d2] = State::get_unused_edges(input, &self.assigns[d2], &dijkstra_d2);
        self.daily_dijkstra[d2] = dijkstra_d2;
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

    fn get_unused_edges(input: &Input, assign: &HashSet<usize>, dijkstras: &Vec<Dijkstra>) -> HashSet<usize>{
        let mut unused_edges: HashSet<usize> = (0..input.M).collect();
        for dijkstra in dijkstras.iter(){
            for e in dijkstra.used_edges.iter(){
                // TODO:　まとめて作ったほうがいいかも
                unused_edges.remove(e);
            }
        }
        for e in assign.iter(){
            unused_edges.remove(e);
        }
        unused_edges
    }

    fn get_daily_dijkstra(input: &Input, removed_edges: &HashSet<usize>, sample_nodes: &HashSet<usize>) -> Vec<Dijkstra>{
        let mut dijkstras = vec![];
        for u in sample_nodes.iter(){
            let dijkstra = Dijkstra::new(&input.G, *u, None, removed_edges);
            dijkstras.push(dijkstra);
        }
        dijkstras
    }

    fn get_daily_score_with_sample_nodes(input: &Input, removed_edges: &HashSet<usize>, sample_nodes: &HashSet<usize>) -> usize{
        State::get_daily_dijkstra(input, removed_edges, sample_nodes).iter().map(|x| x.total_dist).sum()
    }

    fn get_daily_score_from_dijkstras(dijkstras: &Vec<Dijkstra>) -> usize{
        dijkstras.iter().map(|dij| dij.total_dist).sum()
    }

    fn get_daily_score__(input: &Input, removed_edges: &HashSet<usize>) -> usize{
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
            score += State::get_daily_score__(input, &self.assigns[d]);
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

fn climbing(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    eprintln!("climbing start at {}", timer.elapsed().as_secs_f64());
    while timer.elapsed().as_secs_f64()<limit{
        let (mode, params) = state.get_neighbor(input);
        if mode=="none"{
            continue;
        }
        all_iter += 1;
        let (d1, u1, d2, u2) = params;
        let pre_score_d1 = state.daily_score[d1];
        let pre_score_d2 = state.daily_score[d2];
        state.update(&mode, params);
        let aft_dijkstra_d1 = State::get_daily_dijkstra(input, &state.assigns[d1], &state.sample_nodes);
        let aft_dijkstra_d2 = State::get_daily_dijkstra(input, &state.assigns[d2], &state.sample_nodes);
        let aft_score_d1 = State::get_daily_score_from_dijkstras(&aft_dijkstra_d1);
        let aft_score_d2 = State::get_daily_score_from_dijkstras(&aft_dijkstra_d2);
        state.undo(&mode, params);
        let score_diff:i128 = ((aft_score_d1+aft_score_d2) as i128)-((pre_score_d1+pre_score_d2) as i128);
        if score_diff<=0{
            eprintln!("{} {:?} {}", mode, params, score_diff);
            accepted_cnt += 1;
            state.update(&mode, params);
            state.update_with_score(input, &mode, params, (aft_dijkstra_d1, aft_dijkstra_d2));
        }
    }
    eprintln!("===== climbing =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("");
    state
}


fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    let state = solve(&input, &timer);
    println!("{}", &state);
    eprintln!("end at {}", timer.elapsed().as_secs_f64());
}


fn solve(input: &Input, timer:&Instant) -> State {
    let init_state = State::init(input, SEED);
    climbing(input, init_state, timer, 58.0)
}
