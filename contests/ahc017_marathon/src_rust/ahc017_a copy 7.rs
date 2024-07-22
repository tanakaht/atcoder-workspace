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
const DEFAULTDIST: usize = 1000000000;
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
    total_dist: usize,
}

impl Dijkstra {
    fn new(G: &Vec<Vec<(usize, usize, usize)>>, start_node: usize, end_node: Option<usize>, removed_edge: &HashSet<usize>) -> Self {
        let N = G.len();
        let mut q: BinaryHeap<(usize, usize)> = BinaryHeap::new();
        let mut dists = vec![DEFAULTDIST; N];
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
                let d2 = d+c;
                if(dists[*v]<=d2 || removed_edge.contains(i)){
                    continue;
                }
                dists[*v] = d2;
                q.push((INF-d2, *v));
            }
        }
        let total_dist = dists.iter().sum();
        Self { dists, start_node, end_node, total_dist }
    }

    fn distance(&self, target: usize) -> usize {
        self.dists[target]
    }

    fn recalculate(&mut self, input: &Input, removed_edge: &HashSet<usize>, unremoved_edge: &Vec<usize>, additional_remove_edge: &Vec<usize>){
        let mut q: BinaryHeap<(usize, usize)> = BinaryHeap::new();
        let mut appeared = vec![false; input.N];
        // popについてdistとqを定義
        // targetを取ってくる
        let mut target_node: HashSet<usize> = HashSet::new();
        for i in additional_remove_edge.iter(){
            let mut q_dfs: BinaryHeap<usize>  = BinaryHeap::new();
            let (u, v, w) = input.UVW[*i];
            if self.dists[u]+w==self.dists[v]{
                q_dfs.push(v);
            }
            if self.dists[v]+w==self.dists[u]{
                q_dfs.push(u);
            }
            while let Some(node) = q_dfs.pop(){
                target_node.insert(node);
                for (node_, c, i) in input.G[node].iter(){
                    if (self.dists[node]+c==self.dists[*node_] && !removed_edge.contains(i)){
                        q_dfs.push(*node_);
                    }
                }
            }
            eprintln!("{:?} {:?}", target_node, (u, v));
        }
        // dists初期化
        for u in target_node.iter(){
            self.dists[*u] = DEFAULTDIST;
        }
        // qに追加
        for u in target_node.iter(){
            for (v, c, i) in input.G[*u].iter(){
                if target_node.contains(v){
                    continue;
                }
                eprintln!("node found {} {} {} {} {}", u, v, self.dists[*u], self.dists[*v], removed_edge.contains(i));
                let d = self.dists[*v]+c;
                if (self.dists[*u]<=d || removed_edge.contains(i)){
                    continue;
                }
                self.dists[*u] = d;
                q.push((INF-d, *u));
            }
        }
        // addについて　distとqを定義
        for i in unremoved_edge.iter(){
            let (u, v, w) = input.UVW[*i];
            if self.dists[u]>self.dists[v]+w{
                let d = self.dists[v]+w;
                self.dists[u] = d;
                q.push((INF-d, u));
            } else if self.dists[v]>self.dists[u]+w{
                let d = self.dists[u]+w;
                self.dists[v] = d;
                q.push((INF-d, v));
            }
        }
        eprintln!("que: {:?}", q);
        // 差分だけdijkstra
        while let Some((d_, u)) = q.pop() {
            let d = INF-d_;
            if appeared[u]{
                continue;
            }
            appeared[u] = true;
            for (v, c, i) in input.G[u].iter(){
                let d2 = d+c;
                if(self.dists[*v]<=d2 || removed_edge.contains(i)){
                    continue;
                }
                self.dists[*v] = d2;
                q.push((INF-d2, *v));
            }
        }
        self.total_dist = self.dists.iter().sum();
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
    daily_dijkstra: Vec<Vec<Dijkstra>>
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
        // daily_score, daily_dijkstra
        let mut daily_dijkstra: Vec<Vec<Dijkstra>> = vec![];
        let mut daily_score: Vec<usize> = vec![0; input.D];
        for d in 0..input.D{
            let mut dijkstras: Vec<Dijkstra> = vec![];
            for u in sample_nodes.iter(){
                let dijkstra = Dijkstra::new(&input.G, *u, None, &assigns[d]);
                daily_score[d] += dijkstra.total_dist;
                dijkstras.push(dijkstra);
            }
            daily_dijkstra.push(dijkstras);
        }
        Self { assigns, ans, sample_nodes, daily_score, daily_dijkstra }
    }

    fn get_neighbor(&self, input: &Input) -> (String, (usize, usize, usize, usize)){
        let mut rng = rand::thread_rng();
        if rng.gen::<usize>()%10 < 8{
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
        } else {
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

    fn update(&mut self, input: &Input, mode: &str, params: (usize, usize, usize, usize)){
        if mode=="swap"{
            let (d1, u1, d2, u2) = params;
            self.assigns[d1].remove(&u1);
            self.assigns[d2].remove(&u2);
            self.assigns[d1].insert(u2);
            self.assigns[d2].insert(u1);
            self.ans[u1] = d2;
            self.ans[u2] = d1;
            self.daily_score[d1] = 0;
            self.daily_score[d2] = 0;
            for dijkstra in self.daily_dijkstra[d1].iter_mut(){
                dijkstra.recalculate(input, &self.assigns[d1], &vec![u1], &vec![u2]);
                self.daily_score[d1] += dijkstra.total_dist;
            }
            for dijkstra in self.daily_dijkstra[d2].iter_mut(){
                dijkstra.recalculate(input, &self.assigns[d2], &vec![u2], &vec![u1]);
                self.daily_score[d2] += dijkstra.total_dist;
            }
        } else if mode=="move"{
            let (d1, u1, d2, u2) = params;
            self.assigns[d1].remove(&u1);
            self.assigns[d2].insert(u1);
            self.ans[u1] = d2;
            self.daily_score[d1] = 0;
            self.daily_score[d2] = 0;
            for (i, dijkstra) in self.daily_dijkstra[d1].iter_mut().enumerate(){
                dijkstra.recalculate(input, &self.assigns[d1], &vec![u1], &vec![]);
                self.daily_score[d1] += dijkstra.total_dist;
                assert_eq!(dijkstra.dists, Dijkstra::new(&input.G, dijkstra.start_node, None, &self.assigns[d1]).dists);
            }
            for dijkstra in self.daily_dijkstra[d2].iter_mut(){
                dijkstra.recalculate(input, &self.assigns[d2], &vec![], &vec![u1]);
                self.daily_score[d2] += dijkstra.total_dist;
            }
        }
    }

    fn undo(&mut self, input: &Input, mode: &str, params: (usize, usize, usize, usize)){
        if mode=="swap"{
            let (d1, u1, d2, u2) = params;
            self.update(input, mode, (d1, u2, d2, u1));
        } else if mode=="move"{
            let (d1, u1, d2, u2) = params;
            self.update(input, mode, (d2, u1, d1, u2));
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

fn climbing(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    eprintln!("climbing start at {}", timer.elapsed().as_secs_f64());
    while timer.elapsed().as_secs_f64()<limit{
        all_iter += 1;
        let (mode, params) = state.get_neighbor(input);
        let (d1, u1, d2, u2) = params;
        let pre_score_d1 = state.daily_score[d1];
        let pre_score_d2 = state.daily_score[d2];
        state.update(input, &mode, params);
        let aft_score_d1 = state.daily_score[d1];
        let aft_score_d2 = state.daily_score[d2];
        let score_diff:i128 = ((aft_score_d1+aft_score_d2) as i128)-((pre_score_d1+pre_score_d2) as i128);
        if score_diff<=0{
            accepted_cnt += 1;
            state.daily_score[d1] = aft_score_d1;
            state.daily_score[d2] = aft_score_d2;
        } else {
            state.undo(input, &mode, params);
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
    let mut dij = init_state.daily_dijkstra[0][0].clone();
    let start_node = dij.start_node;
    let mut e = input.G[start_node][0].2;
    let mut removed_edge = init_state.assigns[0].clone();
    removed_edge.remove(&e);
    dij.recalculate(input, &removed_edge, &vec![], &vec![e]);
    let dij2 = Dijkstra::new(&input.G, dij.start_node, None, &removed_edge);
    eprintln!("{} {:?} {}", e, input.UVW[e], start_node);
    for u in 0..input.N{
        assert_eq!((u, dij.distance(u)), (u, dij2.distance(u)));
    }
    climbing(input, init_state, timer, 5.8)
}
