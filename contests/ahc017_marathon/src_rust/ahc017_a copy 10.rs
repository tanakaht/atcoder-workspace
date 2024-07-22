#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::HashSet;
use itertools::Itertools;
use num::{range, ToPrimitive};
use proconio::*;
use std::iter::FromIterator;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fmt::Display;
use std::cmp::Reverse;
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
        (((self.x as f64)-(p.y as f64)).powf(2.0)+((self.y as f64)-(p.y as f64)).powf(2.0)) as usize
    }
    fn theta(&self, p: &Point) -> f64{
        ((self.y as f64)-(p.y as f64)).atan2((self.x as f64)-(p.x as f64))
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
        let mut q: BinaryHeap<(Reverse<usize>, usize)> = BinaryHeap::new();
        let mut dists = vec![DEFAULTDIST; N];
        let mut appeared = vec![false; N];
        dists[start_node] = 0;
        q.push((Reverse(0), start_node));
        while let Some((Reverse(d), u)) = q.pop() {
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
                q.push((Reverse(d2), *v));
            }
        }
        let total_dist = dists.iter().sum();
        Self { dists, start_node, end_node, total_dist }
    }

    fn distance(&self, target: usize) -> usize {
        self.dists[target]
    }

    fn recalculate(&mut self, input: &Input, removed_edge: &HashSet<usize>, unremoved_edge: &Vec<usize>, additional_remove_edge: &Vec<usize>){
        let mut q: BinaryHeap<(Reverse<usize>, usize)> = BinaryHeap::new();
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
        }
        // dists初期化
        for u in target_node.iter(){
            self.total_dist += DEFAULTDIST-self.dists[*u];
            self.dists[*u] = DEFAULTDIST;
        }
        // qに追加
        for u in target_node.iter(){
            for (v, c, i) in input.G[*u].iter(){
                if target_node.contains(v){
                    continue;
                }
                let d = self.dists[*v]+c;
                if (self.dists[*u]<=d || removed_edge.contains(i)){
                    continue;
                }
                self.total_dist += d-self.dists[*u];
                self.dists[*u] = d;
                q.push((Reverse(d), *u));
            }
        }
        // addについて　distとqを定義
        for i in unremoved_edge.iter(){
            let (u, v, w) = input.UVW[*i];
            if self.dists[u]>self.dists[v]+w{
                let d = self.dists[v]+w;
                self.total_dist += d-self.dists[u];
                self.dists[u] = d;
                q.push((Reverse(d), u));
            } else if self.dists[v]>self.dists[u]+w{
                let d = self.dists[u]+w;
                self.total_dist += d-self.dists[v];
                self.dists[v] = d;
                q.push((Reverse(d), v));
            }
        }
        //更新あるかチェック
        if (target_node.is_empty() && q.is_empty()){
            return;
        }
        // 差分だけdijkstra
        while let Some((Reverse(d), u)) = q.pop() {
            if appeared[u]{
                continue;
            }
            appeared[u] = true;
            for (v, c, i) in input.G[u].iter(){
                let d2 = d+c;
                if(self.dists[*v]<=d2 || removed_edge.contains(i)){
                    continue;
                }
                self.total_dist += d2-self.dists[*v];
                self.dists[*v] = d2;
                q.push((Reverse(d2), *v));
            }
        }
        // self.total_dist = self.dists.iter().sum();
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
        let mut sample_nodes_ = vec![0; 10];
        let mut XY_enumerate: Vec<(usize, &Point)> = input.XY.iter().enumerate().into_iter().collect();
        let center = Point{ x: 500, y:500};
        XY_enumerate.shuffle(&mut rng);
        for (u, p) in input.XY.iter().enumerate().sorted_by_key(|(ai, ap)| input.G[*ai].len()){
            if p.dist(&center)<12500 {
                sample_nodes_[rng.gen::<usize>()%2] = u;
            }
            else{
                //let idx = 2+(((2.0*(p.theta(&center)+2.0*PI)/PI)%4.0) as usize)*2+(rng.gen::<usize>()%2);
                let idx = 2+(((4.0*(p.theta(&center)+2.0*PI)/PI)%8.0) as usize);
                sample_nodes_[idx] = u;
            }
        }
        let sample_nodes = HashSet::from_iter(sample_nodes_);
        // let v:Vec<usize> = (0..input.N).collect();
        // let sample_nodes: HashSet<usize> = HashSet::from_iter(v.choose_multiple(&mut rng, 10).cloned().collect_vec());
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

    fn get_neighbor(&self, input: &Input) -> (usize, Vec<usize>, usize, Vec<usize>){
        let mut rng = rand::thread_rng();
        let flg = rng.gen::<usize>()%100;
        if flg < 5{
            for _ in 0..100{
                let d1 = rng.gen::<usize>()%input.D;
                let d2 = rng.gen::<usize>()%input.D;
                if (d1==d2 || self.assigns[d1].is_empty() || self.assigns[d2].is_empty()){
                    continue;
                }
                let u1 = *self.assigns[d1].iter().choose(&mut rng).unwrap();
                let u2 = *self.assigns[d2].iter().choose(&mut rng).unwrap();
                let mut us1: Vec<usize> = vec![u1];
                let mut us2: Vec<usize> = vec![u2];
                // dfsしながら追加する
                let mut cur = input.UVW[u1].0;
                if rng.gen::<usize>()%2<1{
                    cur = input.UVW[u1].1;
                }
                loop {
                    let mut found = false;
                    for (v, w, i) in input.G[cur].iter(){
                        if (self.assigns[d1].contains(i) && rng.gen::<usize>()%100<80 && !us1.contains(i)){
                            cur = *v;
                            us1.push(*i);
                            found = true;
                            break;
                        }
                    }
                    if !found{
                        break
                    }
                }
                cur = input.UVW[u2].0;
                if rng.gen::<usize>()%2<1{
                    cur = input.UVW[u2].1;
                }
                loop {
                    let mut found = false;
                    for (v, w, i) in input.G[cur].iter(){
                        if (self.assigns[d2].contains(i) && rng.gen::<usize>()%100<80 && !us2.contains(i)){
                            cur = *v;
                            us2.push(*i);
                            found = true;
                            break;
                        }
                    }
                    if !found{
                        break
                    }
                }
                if ( self.assigns[d1].len()+us2.len()-us1.len()>input.K || self.assigns[d2].len()+us1.len()-us2.len()>input.K ) {
                    continue;
                }
                return (d1, us1, d2, us2);
            }
        // } else if flg < 5{
        //     for _ in 0..100{
        //         let d1 = rng.gen::<usize>()%input.D;
        //         let d2 = rng.gen::<usize>()%input.D;
        //         if (d1==d2 || self.assigns[d1].is_empty() || self.assigns[d2].is_empty()){
        //             continue;
        //         }
        //         let u1 = *self.assigns[d1].iter().choose(&mut rng).unwrap();
        //         let mut us1: Vec<usize> = vec![u1];
        //         let mut us2: Vec<usize> = vec![];
        //         // dfsしながら追加する
        //         let mut cur = input.UVW[u1].0;
        //         loop {
        //             let mut found = false;
        //             for (v, w, i) in input.G[cur].iter(){
        //                 if (self.assigns[d1].contains(i) && rng.gen::<usize>()%100<80 && !us1.contains(i)){
        //                     cur = *v;
        //                     us1.push(*i);
        //                     found = true;
        //                     break;
        //                 }
        //             }
        //             if !found{
        //                 break
        //             }
        //         }
        //         if ( self.assigns[d1].len()+us2.len()-us1.len()>input.K || self.assigns[d2].len()+us1.len()-us2.len()>input.K ) {
        //             continue;
        //         }
        //         return (d1, us1, d2, us2);
        //     }
        // } else if flg < 5{
        //     for _ in 0..100{
        //         let d1 = rng.gen::<usize>()%input.D;
        //         let d2 = rng.gen::<usize>()%input.D;
        //         if (d1==d2 || self.assigns[d1].is_empty() || self.assigns[d2].is_empty()){
        //             continue;
        //         }
        //         let u1 = *self.assigns[d1].iter().choose(&mut rng).unwrap();
        //         let u2 = *self.assigns[d2].iter().choose(&mut rng).unwrap();
        //         return (d1, vec![u1], d2, vec![u2]);
        //     }
        } else {
            for _ in 0..100{
                let d1 = rng.gen::<usize>()%input.D;
                let d2 = rng.gen::<usize>()%input.D;
                if (d1==d2 || self.assigns[d1].is_empty() || self.assigns[d2].len()>=input.K){
                    continue;
                }
                let u1 = *self.assigns[d1].iter().choose(&mut rng).unwrap();
                return (d1, vec![u1], d2, vec![]);
            }
        }
        (0, vec![], 1, vec![])
    }

    fn update(&mut self, input: &Input, params: (usize, Vec<usize>, usize, Vec<usize>)){
        let (d1, us1, d2, us2) = params;
        for u1 in us1.iter(){
            self.assigns[d1].remove(u1);
            self.assigns[d2].insert(*u1);
            self.ans[*u1] = d2;
        }
        for u2 in us2.iter(){
            self.assigns[d2].remove(u2);
            self.assigns[d1].insert(*u2);
            self.ans[*u2] = d1;
        }
        self.daily_score[d1] = 0;
        self.daily_score[d2] = 0;
        for dijkstra in self.daily_dijkstra[d1].iter_mut(){
            dijkstra.recalculate(input, &self.assigns[d1], &us1, &us2);
            self.daily_score[d1] += dijkstra.total_dist;
        }
        for dijkstra in self.daily_dijkstra[d2].iter_mut(){
            dijkstra.recalculate(input, &self.assigns[d2], &us2, &us1);
            self.daily_score[d2] += dijkstra.total_dist;
        }
    }

    fn undo(&mut self, input: &Input, params: (usize, Vec<usize>, usize, Vec<usize>)){
        let (d1, us1, d2, us2) = params;
        self.update(input, (d1, us2, d2, us1));
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
    let mut movecnt = 0;
    let mut swapcnt = 0;
    eprintln!("climbing start at {}", timer.elapsed().as_secs_f64());
    loop {
        let elasped_time = timer.elapsed().as_secs_f64();
        if elasped_time >= limit{
            break;
        }
        let params = state.get_neighbor(input);
        if params.1.len()+params.3.len()==0{
            continue;
        }
        all_iter += 1;
        let pre_score_d1 = state.daily_score[params.0];
        let pre_score_d2 = state.daily_score[params.2];
        state.update(input, params.clone());
        let aft_score_d1 = state.daily_score[params.0];
        let aft_score_d2 = state.daily_score[params.2];
        let score_diff:i128 = ((aft_score_d1+aft_score_d2) as i128)-((pre_score_d1+pre_score_d2) as i128);
        if score_diff<=0{
            accepted_cnt += 1;
            // if mode=="move"{
            //     movecnt += 1;
            // } else {
            //     swapcnt += 1;
            // }
            //eprintln!("{} {} {}", timer.elapsed().as_secs_f32(), movecnt, swapcnt);
        } else {
            state.undo(input, params);
        }
    }
    eprintln!("===== climbing =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("");
    state
}

fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut movecnt = 0;
    let mut swapcnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 10000.0;
    let end_temp: f64 = 0.1;
    let mut temp = start_temp;
    eprintln!("simanneal start at {}", timer.elapsed().as_secs_f64());
    loop {
        let elasped_time = timer.elapsed().as_secs_f64();
        if elasped_time >= limit{
            break;
        }
        all_iter += 1;
        let params = state.get_neighbor(input);
        let pre_score_d1 = state.daily_score[params.0];
        let pre_score_d2 = state.daily_score[params.2];
        state.update(input, params.clone());
        let aft_score_d1 = state.daily_score[params.0];
        let aft_score_d2 = state.daily_score[params.2];
        let score_diff:i128 = ((aft_score_d1+aft_score_d2) as i128)-((pre_score_d1+pre_score_d2) as i128);
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
    if (score_diff<=0 || ((-score_diff as f64)/temp).exp() > rng.gen::<f64>()){
            accepted_cnt += 1;
            // if mode=="move"{
            //     movecnt += 1;
            // } else {
            //     swapcnt += 1;
            // }
            //eprintln!("{} {} {}", timer.elapsed().as_secs_f32(), movecnt, swapcnt);
        } else {
            state.undo(input, params);
        }
    }
    eprintln!("===== simanneal =====");
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
    climbing(input, init_state, timer, 5.8)
}
