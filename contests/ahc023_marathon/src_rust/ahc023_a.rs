#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap, VecDeque};
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

// p2の方が良いならtrue
fn is_good_p2(p1: &(usize, Flg, Vec<usize>, usize), p2: &(usize, Flg, Vec<usize>, usize)) -> bool{
    // (p1.0<p2.0) || (p1.0==p2.0 && p1.3==p2.3 && p1.2.len()>p2.2.len())
    (p1.0<p2.0) || ( p1.0==p2.0 &&  p1.3 > p2.3) || (p1.0==p2.0 && p1.3==p2.3 && p1.2.len()>p2.2.len())
}

// TODO: lowlinkなことはわかっている
fn get_articulation_points(g: &Vec<Vec<usize>>, s: usize) -> (HashSet<usize>, Vec<HashSet<usize>>) {
    let n = g.len();
    let mut articulation_points = HashSet::new();
    let mut art_points_children = vec![HashSet::new(); n];
    for v in 0..n{
        let mut appeared = HashSet::new();
        let mut q = vec![];
        appeared.insert(s);
        q.push(s);
        while !q.is_empty(){
            let idx = q.pop().unwrap();
            for idx2 in g[idx].iter(){
                if *idx2!=v && !appeared.contains(idx2){
                    appeared.insert(*idx2);
                    q.push(*idx2);
                }
            }
        }
        if appeared.len()<n-1{
            articulation_points.insert(v);
            let children: HashSet<usize> = (0..n).filter(|i| !appeared.contains(i)).collect();
            art_points_children[v] =  children;
        }
    }
    (articulation_points, art_points_children)
}


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


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    t: usize,
    h: usize,
    w: usize,
    i0: usize,
    g: Vec<Vec<usize>>,
    edges: Vec<(usize, usize)>,
    art_points: HashSet<usize>,
    art_point_children: Vec<HashSet<usize>>,
    k: usize,
    sd: Vec<(usize, usize)>,
    flgs: Vec<Flg>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            t: usize,
            h: usize,
            w: usize,
            i0: usize,
            H: [String; h-1],
            V: [String; h],
            k: usize,
            sd_: [(usize, usize); k],
        }
        let mut g: Vec<Vec<usize>> = vec![vec![]; h*w];
        let mut edges: Vec<(usize, usize)> = vec![];
        let sd = sd_.iter().map(|&(s, d)| ((s-1)*2, (d-1)*2+1)).collect_vec();
        let flgs = sd_.iter().map(|&(s, d)| Flg::from_invalid((s-1)*2, (d-1)*2+1)).collect_vec();
        for i in 0..(h-1){
            let H_ = usize::from_str_radix(&H[i], 2).unwrap();
            for j in 0..w{
                if (H_>>(w-1-j))&1==0{
                    g[i*w+j].push((i+1)*w+j);
                    g[(i+1)*w+j].push(i*w+j);
                    edges.push((i*w+j, (i+1)*w+j));
                }
            }
        }
        for i in 0..h{
            let V_ = usize::from_str_radix(&V[i], 2).unwrap();
            for j in 0..(w-1){
                if (V_>>(w-2-j))&1==0{
                    g[i*w+j].push(i*w+j+1);
                    g[i*w+j+1].push(i*w+j);
                    edges.push((i*w+j, i*w+j+1));
                }
            }
        }
        let (art_points, art_point_children) = get_articulation_points(&g, i0*w);
        Self { t, h, w, i0, g, edges, art_points, art_point_children, k, sd, flgs }
    }

    fn get_mindist_tree(&self) -> (Vec<Option<usize>>, Vec<Vec<usize>>){
        let mut parents = vec![None; self.h*self.w];
        let mut children = vec![vec![]; self.h*self.w];
        let mut q: VecDeque<(usize, Option<usize>)> = VecDeque::new();
        let mut appeared = vec![false; self.h*self.w];
        q.push_back((self.i0*self.w, None));
        while !q.is_empty(){
            let (idx, parent) = q.pop_front().unwrap();
            if appeared[idx]{
                continue;
            }
            appeared[idx] = true;
            if let Some(x) = parent{
                children[x].push(idx);
            }
            parents[idx] = parent;
            for idx2 in self.g[idx].iter(){
                if !appeared[*idx2]{
                    q.push_back((*idx2, Some(idx)));
                }
            }
        }
        (parents, children)
    }

    fn get_random_tree(&self) -> (Vec<Option<usize>>, Vec<Vec<usize>>){
        let mut parents = vec![None; self.h*self.w];
        let mut children = vec![vec![]; self.h*self.w];
        let mut uf = UnionFind::new(self.h*self.w);
        let mut edges = self.edges.clone();
        let mut g = vec![vec![]; self.h*self.w];
        edges.shuffle(&mut rand::thread_rng());
        for &(i, j) in edges.iter(){
            if !uf.same(i, j){
                uf.union(i, j);
                g[i].push(j);
                g[j].push(i);
            }
        }
        let mut q = vec![];
        q.push((self.i0*self.w, None));
        while !q.is_empty(){
            let (idx, parent) = q.pop().unwrap();
            parents[idx] = parent;
            for &idx2 in g[idx].iter(){
                if Some(idx2)!=parent{
                    q.push((idx2, Some(idx)));
                    children[idx].push(idx2);
                }
            }
        }
        (parents, children)
    }

    fn get_subtree_idxs_til_art_point(&self, childrens: &Vec<Vec<usize>>, subtree_root: usize) -> Vec<usize>{
        // TODO: bfsがいい?dfsがいい?
        let mut subtree_idxs = vec![];
        let mut q = VecDeque::new();
        q.push_back(subtree_root);
        while !q.is_empty(){
            let idx = q.pop_front().unwrap();
            subtree_idxs.push(idx);
            if self.art_points.contains(&idx){
                continue;
            }
            for &idx2 in childrens[idx].iter(){
                q.push_back(idx2);
            }
        }
        subtree_idxs
    }

    fn get_subtree_idxs(&self, childrens: &Vec<Vec<usize>>, subtree_root: usize) -> Vec<usize>{
        // TODO: bfsがいい?dfsがいい?
        let mut subtree_idxs = vec![];
        let mut q = VecDeque::new();
        q.push_back(subtree_root);
        while !q.is_empty(){
            let idx = q.pop_front().unwrap();
            subtree_idxs.push(idx);
            for &idx2 in childrens[idx].iter(){
                q.push_back(idx2);
            }
        }
        subtree_idxs
    }

    fn get_ancestors_til_art_point(&self, parents: &Vec<Option<usize>>, idx: usize) -> Vec<usize>{
        let mut ancestors = vec![];
        let mut cur = idx;
        while let Some(cur_) = parents[cur]{
            ancestors.push(cur_);
            if self.art_points.contains(&cur_){
                break;
            }
            cur = cur_;
        }
        ancestors
    }

    fn get_ancestors(&self, parents: &Vec<Option<usize>>, idx: usize) -> Vec<usize>{
        let mut ancestors = vec![];
        let mut cur = idx;
        while let Some(cur_) = parents[cur]{
            ancestors.push(cur_);
            cur = cur_;
        }
        ancestors
    }
}



#[allow(unused_variables)]
#[derive(Debug, Clone, Eq, PartialEq)]
struct Flg{
    bit0: u128,
    bit1: u128,
}

impl Flg{
    fn one() -> Self{
        Self{bit0: (1<<100)-1, bit1: (1<<100)-1}
    }

    fn zero() -> Self{
        Self{bit0: 0, bit1: 0}
    }

    fn popcnt(&self) -> usize{
        self.bit0.count_ones() as usize + self.bit1.count_ones() as usize
    }

    // (l, r)がvalid
    fn from_valid(l: usize, r:usize) -> Self{
        if l<100{
            if r<100{
                Self{bit0: ((1<<r)-1)^((1<<(l+1))-1), bit1: 0}
            } else {
                Self{bit0: ((1<<100)-1)^((1<<(l+1))-1), bit1: ((1<<(r-100))-1)}
            }
        } else {
            Self{bit0: 0, bit1: ((1<<(r-100))-1)^((1<<(l-100+1))-1)}
        }
    }

    // (l, r)がinvalid
    fn from_invalid(l: usize, r:usize) -> Self{
        if l<100{
            if r<100{
                Self{bit0: ((1<<100)-1)^((1<<r)-1)^((1<<(l+1))-1), bit1: (1<<100)-1}
            } else {
                Self{bit0: (1<<(l+1))-1, bit1: ((1<<100)-1)^((1<<(r-100))-1)}
            }
        } else {
            Self{bit0: (1<<100)-1, bit1: ((1<<100)-1)^((1<<(r-100))-1)^((1<<(l-100+1))-1)}
        }
    }

    // [l, r)がinvalid
    fn from_invalid2(l: usize, r:usize) -> Self{
        if l<100{
            if r<100{
                Self{bit0: ((1<<100)-1)^((1<<r)-1)^((1<<(l))-1), bit1: (1<<100)-1}
            } else {
                Self{bit0: (1<<(l))-1, bit1: ((1<<100)-1)^((1<<(r-100))-1)}
            }
        } else {
            Self{bit0: (1<<100)-1, bit1: ((1<<100)-1)^((1<<(r-100))-1)^((1<<(l-100))-1)}
        }
    }

    fn from_single(i: usize) -> Self{
        if i<100{
            Self{bit0: 1<<i, bit1: 0}
        } else {
            Self{bit0: 0, bit1: 1<<(i-100)}
        }
    }

    fn is_ok(&self, i: usize)->bool{
        if i<100{
            (self.bit0>>i)&1==1
        } else {
            (self.bit1>>(i-100))&1==1
        }
    }

    fn or(&self, other: &Flg) -> Flg{
        Self { bit0: self.bit0|other.bit0, bit1: self.bit1|other.bit1 }
    }

    fn and(&self, other: &Flg) -> Flg{
        Self { bit0: self.bit0&other.bit0, bit1: self.bit1&other.bit1 }
    }

    fn eq(&self, other: &Flg) -> bool{
        self.bit0==other.bit0 && self.bit1==other.bit1
    }

    fn is_bigger(&self, other: &Flg) -> bool{
        (self.bit0&other.bit0)>=self.bit0 && (self.bit1&other.bit1)>=self.bit1
    }

    // fn min_bit(&self, from: usize) -> usize{
    // }

    fn max_bit(&self, from: usize) -> Option<usize>{
        let flg = if (from<100){ self.and(&Self{bit0: (1<<(from+1))-1, bit1: 0}) } else { self.and(&Self{bit0: (1<<100)-1, bit1: (1<<(from-100+1))-1}) };
        if flg.bit1>0{
            let bit = flg.bit1;
            let mut l= 0;
            let mut r= 100;
            // [l, r)にbitが立っている
            while r-l>1{
                let m = (l+r)/2;
                if bit&((1<<m)-1)==bit{
                    r = m;
                } else {
                    l = m;
                }
            }
            Some(l+100)
        } else if flg.bit0>0{
            let bit = flg.bit0;
            let mut l= 0;
            let mut r= 100;
            // [l, r)にbitが立っている
            while r-l>1{
                let m = (l+r)/2;
                if bit&((1<<m)-1)==bit{
                    r = m;
                } else {
                    l = m;
                }
            }
            Some(l)
        } else {
            None
        }
    }
}


impl PartialOrd for Flg {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.popcnt().cmp(&other.popcnt()))
    }
}


impl Ord for Flg {
    fn cmp(&self, other: &Self) -> Ordering {
        self.popcnt().cmp(&other.popcnt())
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct NeighborParam{
    moves: Vec<(usize, usize, usize)> // i番目の作物をjからkに移動
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    places: Vec<usize>,
    place2idx: Vec<HashSet<usize>>,
    place2score: Vec<usize>,
    rest_sdi: Vec<Vec<(usize, usize)>>
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut place2idx = vec![HashSet::new(); input.h*input.w+1];
        for i in 0..input.k{
            place2idx[input.h*input.w].insert(i);
        }
        let mut rest_sdi = vec![vec![]; input.t*2];
        for (i, (s, d)) in input.sd.iter().enumerate(){
            rest_sdi[*s].push((*d, i));
        }
        for s in 0..(input.t*2){
            rest_sdi[s].sort_by_key(|&(d, _)| d);
        }
        let mut state = Self{places: vec![input.h*input.w; input.k], place2idx, place2score: vec![0; input.h*input.w], rest_sdi};
        // moveを作る
        let (parents, childrens) = input.get_mindist_tree();
        state.optimize_subtree(input, input.i0*input.w, &parents, &childrens);
        // state.update(&neighborparam, input);
        state
    }

    fn get_mapflg(&self, input: &Input, idx: usize) -> Flg{
        self.place2idx[idx].iter().fold(Flg::one(), |acc, &i| acc.and(&input.flgs[i]))
    }

    fn get_reachmap(&self, input: &Input) -> Vec<Flg>{
        let mut flgs = vec![Flg::zero(); input.h*input.w];  // 入った時に更新
        let mut last_updated = vec![Flg::zero(); input.h*input.w];  // q入れた時に更新
        // let mut q = VecDeque::new();
        let mut q: BinaryHeap<(Flg, usize)> = BinaryHeap::new();
        q.push((self.get_mapflg(input, input.i0*input.w), input.i0*input.w));
        while !q.is_empty(){
            let (flg, idx) = q.pop().unwrap();
            if flg.eq(&flgs[idx]) || !flgs[idx].is_bigger(&flg){
                continue;
            }
            flgs[idx] = flg.clone();
            for idx2 in input.g[idx].iter(){
                let flg2 = flg.and(&self.get_mapflg(input, *idx2)).or(&flgs[*idx2]).or(&last_updated[*idx2]);
                if !flg2.eq(&last_updated[*idx2]) && last_updated[*idx2].is_bigger(&flg2){
                    last_updated[*idx2] = flg2.clone();
                    q.push((flg2, *idx2));
                }
            }
        }
        flgs
    }

    fn get_reachmap_with_target(&self, input: &Input, target: usize, from: (Flg, usize)) -> Vec<Flg>{
        let mut flgs = vec![Flg::zero(); input.h*input.w];  // 入った時に更新
        let mut last_updated = vec![Flg::zero(); input.h*input.w];  // q入れた時に更新
        let mut q: BinaryHeap<(Flg, usize)> = BinaryHeap::new();
        q.push(from);
        // q.push((self.get_mapflg(input, input.i0*input.w), input.i0*input.w));
        while !q.is_empty(){
            let (flg, idx) = q.pop().unwrap();
            if flg.eq(&flgs[idx]) || !flgs[idx].is_bigger(&flg){
                continue;
            }
            flgs[idx] = flg.clone();
            for idx2 in input.g[idx].iter(){
                let flg2 = flg.and(&self.get_mapflg(input, *idx2)).or(&flgs[*idx2]).or(&last_updated[*idx2]);
                if (!flg2.eq(&last_updated[*idx2]) && last_updated[*idx2].is_bigger(&flg2)) && !(input.art_points.contains(&idx) && !input.art_point_children[idx].contains(&target) && input.art_point_children[idx].contains(idx2)){
                    last_updated[*idx2] = flg2.clone();
                    q.push((flg2, *idx2));
                }
            }
        }
        flgs
    }

    // (invalid使わないときのreachmap, invalid使うときのreachmap)
    fn get_reachmap_with_invalid_with_target(&self, input: &Input, invalid: usize, from: (Flg, usize)) -> (Vec<Flg>, Vec<Flg>){
        let mut flgs = vec![Flg::zero(); input.h*input.w];  // 入った時に更新
        let mut last_updated = vec![Flg::zero(); input.h*input.w];  // q入れた時に更新
        // let mut q = VecDeque::new();
        let mut q: BinaryHeap<(Flg, usize)> = BinaryHeap::new();
        q.push((self.get_mapflg(input, input.i0*input.w), input.i0*input.w));
        // q.push(from);
        while !q.is_empty(){
            let (flg, idx) = q.pop().unwrap();
            if flg.eq(&flgs[idx]) || !flgs[idx].is_bigger(&flg){
                continue;
            }
            flgs[idx] = flg.clone();
            if idx==invalid{
                continue;
            }
            for idx2 in input.g[idx].iter(){
                let flg2 = flg.and(&self.get_mapflg(input, *idx2)).or(&flgs[*idx2]).or(&last_updated[*idx2]);
                if (!flg2.eq(&last_updated[*idx2]) && last_updated[*idx2].is_bigger(&flg2)) && (!input.art_points.contains(&idx) || input.art_point_children[idx].contains(&invalid) || !input.art_point_children[idx].contains(idx2)){
                // if (!flg2.eq(&last_updated[*idx2]) && last_updated[*idx2].is_bigger(&flg2)){
                    last_updated[*idx2] = flg2.clone();
                    q.push((flg2, *idx2));
                };
            }
        }
        let mut flgs_with_valid = flgs.clone();
        q.push((flgs_with_valid[invalid].clone(), invalid));
        flgs_with_valid[invalid] = Flg::zero();
        while !q.is_empty(){
            let (flg, idx) = q.pop().unwrap();
            if flg.eq(&flgs_with_valid[idx]) || !flgs_with_valid[idx].is_bigger(&flg){
                continue;
            }
            flgs_with_valid[idx] = flg.clone();
            for idx2 in input.g[idx].iter(){
                let flg2 = flg.and(&self.get_mapflg(input, *idx2)).or(&flgs_with_valid[*idx2]).or(&last_updated[*idx2]);
                if (!flg2.eq(&last_updated[*idx2]) && last_updated[*idx2].is_bigger(&flg2)) && (!input.art_points.contains(&idx) || input.art_point_children[idx].contains(&invalid) || !input.art_point_children[idx].contains(idx2)){
                // if (!flg2.eq(&last_updated[*idx2]) && last_updated[*idx2].is_bigger(&flg2)){
                    last_updated[*idx2] = flg2.clone();
                    q.push((flg2, *idx2));
                };
            }
        }
        (flgs, flgs_with_valid)
    }

    // (invalid使わないときのreachmap, invalid使うときのreachmap)
    fn get_reachmap_with_invalid(&self, input: &Input, invalid: usize) -> (Vec<Flg>, Vec<Flg>){
        let mut flgs = vec![Flg::zero(); input.h*input.w];  // 入った時に更新
        let mut last_updated = vec![Flg::zero(); input.h*input.w];  // q入れた時に更新
        // let mut q = VecDeque::new();
        let mut q: BinaryHeap<(Flg, usize)> = BinaryHeap::new();
        q.push((self.get_mapflg(input, input.i0*input.w), input.i0*input.w));
        while !q.is_empty(){
            let (flg, idx) = q.pop().unwrap();
            if flg.eq(&flgs[idx]) || !flgs[idx].is_bigger(&flg){
                continue;
            }
            flgs[idx] = flg.clone();
            if idx==invalid{
                continue;
            }
            for idx2 in input.g[idx].iter(){
                let flg2 = flg.and(&self.get_mapflg(input, *idx2)).or(&flgs[*idx2]).or(&last_updated[*idx2]);
                if !flg2.eq(&last_updated[*idx2]) && last_updated[*idx2].is_bigger(&flg2){
                    last_updated[*idx2] = flg2.clone();
                    q.push((flg2, *idx2));
                };
            }
        }
        let mut flgs_with_valid = flgs.clone();
        q.push((flgs_with_valid[invalid].clone(), invalid));
        flgs_with_valid[invalid] = Flg::zero();
        while !q.is_empty(){
            let (flg, idx) = q.pop().unwrap();
            if flg.eq(&flgs_with_valid[idx]) || !flgs_with_valid[idx].is_bigger(&flg){
                continue;
            }
            flgs_with_valid[idx] = flg.clone();
            for idx2 in input.g[idx].iter(){
                let flg2 = flg.and(&self.get_mapflg(input, *idx2)).or(&flgs_with_valid[*idx2]).or(&last_updated[*idx2]);
                if !flg2.eq(&last_updated[*idx2]) && last_updated[*idx2].is_bigger(&flg2){
                    last_updated[*idx2] = flg2.clone();
                    q.push((flg2, *idx2));
                };
            }
        }
        (flgs, flgs_with_valid)
    }

    fn is_ok(&self, input: &Input) -> bool{
        for (i, idxs_) in self.place2idx.iter().enumerate(){
            if i==input.h*input.w{
                continue;
            }
            let mut idxs = idxs_.iter().map(|&c| c).collect_vec();
            idxs.sort_by_key(|&i_| input.sd[i_].1);
            let mut flg = Flg::one();
            let mut last_d = 0;
            for idx in idxs.iter(){
                let (s, d) = input.sd[*idx];
                flg = flg.and(&input.flgs[*idx]);
                if let Some(s_) = flg.max_bit(s){
                    if s_<last_d{
                        return false;
                    }
                    last_d = d;
                } else {
                    return false;
                }
            }
        }

        let mut flgs = self.get_reachmap(input);
        for (i, flg_) in flgs.iter().enumerate(){
            let mut idxs = self.place2idx[i].iter().map(|&c| c).collect_vec();
            let mut flg = flg_.clone();
            idxs.sort_by_key(|&i_| input.sd[i_].1);
            for idx in idxs.iter(){
                let (s, d) = input.sd[*idx];
                if flg.max_bit(s).is_none() || !flg.is_ok(d){
                    return false;
                }
                flg = flg.and(&Flg::from_invalid2(0, d));
            }
        }
        true
    }

    fn clear_place(&mut self, input: &Input, idx: usize){
        let mut moves = vec![];
        for i in self.place2idx[idx].iter(){
            moves.push((*i, idx, input.h*input.w));
        }
        self.update(&NeighborParam{moves}, input);
    }

    fn optimize_subtrees(&mut self, input: &Input, subtree_root1: usize, subtree_root2: usize, parents: &Vec<Option<usize>>, childrens: &Vec<Vec<usize>>){
        // まずまっさらにする
        let mut moves = vec![];
        for idx in input.get_subtree_idxs(childrens, subtree_root1){
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        for idx in input.get_ancestors(parents, subtree_root1){
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        for idx in input.get_subtree_idxs(childrens, subtree_root2){
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        for idx in input.get_ancestors(parents, subtree_root2){
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        self.update(&NeighborParam{moves}, input);
        self.optimize_subtree(input, subtree_root1, parents, childrens);
        self.optimize_subtree(input, subtree_root2, parents, childrens);
        self.optimize_ancestors(input, subtree_root1, parents, childrens);
        self.optimize_ancestors(input, subtree_root2, parents, childrens);

    }

    fn optimize_art_point(&mut self, input: &Input, subtree_root: usize, parents: &Vec<Option<usize>>, childrens: &Vec<Vec<usize>>){
        let mut break_point = Some(subtree_root);
        let mut cnt = 0;
        while let Some(idx) = break_point{
            break_point = parents[idx];
            if self.get_single_score(input, idx)==0.0{
                cnt += 1;
            } else {
                cnt = 0;
            }
            if cnt>=2{
                break;
            }
        }
        let mut art_point = subtree_root;
        while input.art_points.contains(&art_point){
            art_point = parents[art_point].unwrap();
        }
        // まずまっさらにする
        let mut moves = vec![];
        for idx in input.get_subtree_idxs_til_art_point(childrens, subtree_root){
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        self.update(&NeighborParam{moves}, input);
        let mut moves = vec![];
        for idx in input.get_ancestors(parents, subtree_root){
            if Some(idx)==break_point{
                break;
            }
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        self.update(&NeighborParam{moves}, input);
        // subtreeより上のshould_openを求めておく。
        let mut flg_art_point = Flg::zero();
        let mut should_opens: Vec<HashSet<usize>> = vec![HashSet::new(); input.h*input.w];
        for &idx in input.get_ancestors(parents, subtree_root).iter(){
            if Some(idx)==break_point{
                break;
            }
            let (flgs_with_invalid, flgs) = self.get_reachmap_with_invalid_with_target(input, idx, (self.get_mapflg(input, input.i0*input.w), input.i0*input.w));
            if idx==art_point{
                flg_art_point = flgs[art_point].clone();
            }
            let should_open = self.get_should_open_with_target(input, idx, flgs_with_invalid, flgs);
            for i in 0..(input.t*2){
                if should_open.is_ok(i){
                    should_opens[idx].insert(i);
                }
            }
        }
        // 葉の方から繰り返し。
        for idx in input.get_subtree_idxs(childrens, subtree_root).iter().rev(){
            // costsを求める。直近の親を重視しつつ、should_openを増やさないようにする。厳密にshould_openは求めず、treeのルートで行く前提。
            let n_ancestors = input.get_ancestors(parents, *idx).len();
            let base_cost = n_ancestors*(n_ancestors+1)/2;
            let mut costs = vec![base_cost; input.t*2];
            for (i, p) in input.get_ancestors(parents, *idx).iter().enumerate(){
                for j in should_opens[*p].iter(){
                    costs[*j] -= i+1;
                }
            }
            // neighbor取得してupdate
            let from_ = if (*idx==art_point){(self.get_mapflg(input, input.i0*input.w), input.i0*input.w)}else{(flg_art_point.clone(), art_point)};
            let neighborparam = self.get_neighbor_optimize_single_place_with_target(input, *idx, &costs, from_);
            self.update(&neighborparam, input);
            for p in input.get_ancestors(parents, *idx).iter(){
                if Some(*p)==break_point{
                    break;
                }
                for idx_ in self.place2idx[*idx].iter(){
                    let (s, d) = input.sd[*idx_];
                    should_opens[*p].insert(s);
                    should_opens[*p].insert(d);
                }
            }
        }
        // ancesctorsについて繰り返し
        for idx in input.get_ancestors(parents, subtree_root).iter(){
            if Some(*idx)==break_point{
                break;
            }
            // costsを求める。直近の親を重視しつつ、should_openを増やさないようにする。厳密にshould_openは求めず、treeのルートで行く前提。
            let n_ancestors = input.get_ancestors(parents, *idx).len();
            let base_cost = n_ancestors*(n_ancestors+1)/2;
            let mut costs = vec![base_cost; input.t*2];
            for (i, p) in input.get_ancestors(parents, *idx).iter().enumerate(){
                for j in should_opens[*p].iter(){
                    costs[*j] -= i+1;
                }
            }
            // neighbor取得してupdate
            let neighborparam = self.get_neighbor_optimize_single_place_with_target(input, *idx, &costs, ((self.get_mapflg(input, input.i0*input.w)), input.i0*input.w));
            self.update(&neighborparam, input);
            for p in input.get_ancestors(parents, *idx).iter(){
                if Some(*p)==break_point{
                    break;
                }
                for idx_ in self.place2idx[*idx].iter(){
                    let (s, d) = input.sd[*idx_];
                    should_opens[*p].insert(s);
                    should_opens[*p].insert(d);
                }
            }
        }
    }


    fn optimize_subtree_and_ancestors(&mut self, input: &Input, subtree_root: usize, parents: &Vec<Option<usize>>, childrens: &Vec<Vec<usize>>){
        let mut break_point = Some(subtree_root);
        let mut cnt = 0;
        while let Some(idx) = break_point{
            break_point = parents[idx];
            if self.get_single_score(input, idx)==0.0{
                cnt += 1;
            } else {
                cnt = 0;
            }
            if cnt>=2{
                break;
            }
        }
        // まずまっさらにする
        let mut moves = vec![];
        for idx in input.get_subtree_idxs(childrens, subtree_root){
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        self.update(&NeighborParam{moves}, input);
        let mut moves = vec![];
        for idx in input.get_ancestors(parents, subtree_root){
            if Some(idx)==break_point{
                break;
            }
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        self.update(&NeighborParam{moves}, input);
        // subtreeより上のshould_openを求めておく。
        let mut should_opens: Vec<HashSet<usize>> = vec![HashSet::new(); input.h*input.w];
        for &idx in input.get_ancestors(parents, subtree_root).iter(){
            if Some(idx)==break_point{
                break;
            }
            let (flgs_with_invalid, flgs) = self.get_reachmap_with_invalid(input, idx);
            let should_open = self.get_should_open(input, idx, flgs_with_invalid, flgs);
            for i in 0..(input.t*2){
                if should_open.is_ok(i){
                    should_opens[idx].insert(i);
                }
            }
        }
        // 葉の方から繰り返し。
        for idx in input.get_subtree_idxs(childrens, subtree_root).iter().rev(){
            // costsを求める。直近の親を重視しつつ、should_openを増やさないようにする。厳密にshould_openは求めず、treeのルートで行く前提。
            let n_ancestors = input.get_ancestors(parents, *idx).len();
            let base_cost = n_ancestors*(n_ancestors+1)/2;
            let mut costs = vec![base_cost; input.t*2];
            for (i, p) in input.get_ancestors(parents, *idx).iter().enumerate(){
                for j in should_opens[*p].iter(){
                    costs[*j] -= i+1;
                }
            }
            // neighbor取得してupdate
            let neighborparam = self.get_neighbor_optimize_single_place_with_target(input, *idx, &costs, ((self.get_mapflg(input, input.i0*input.w)), input.i0*input.w));
            self.update(&neighborparam, input);
            for p in input.get_ancestors(parents, *idx).iter(){
                if Some(*p)==break_point{
                    break;
                }
                for idx_ in self.place2idx[*idx].iter(){
                    let (s, d) = input.sd[*idx_];
                    should_opens[*p].insert(s);
                    should_opens[*p].insert(d);
                }
            }
        }
        // ancesctorsについて繰り返し
        for idx in input.get_ancestors(parents, subtree_root).iter(){
            if Some(*idx)==break_point{
                break;
            }
            // costsを求める。直近の親を重視しつつ、should_openを増やさないようにする。厳密にshould_openは求めず、treeのルートで行く前提。
            let n_ancestors = input.get_ancestors(parents, *idx).len();
            let base_cost = n_ancestors*(n_ancestors+1)/2;
            let mut costs = vec![base_cost; input.t*2];
            for (i, p) in input.get_ancestors(parents, *idx).iter().enumerate(){
                for j in should_opens[*p].iter(){
                    costs[*j] -= i+1;
                }
            }
            // neighbor取得してupdate
            let neighborparam = self.get_neighbor_optimize_single_place_with_target(input, *idx, &costs, ((self.get_mapflg(input, input.i0*input.w)), input.i0*input.w));
            self.update(&neighborparam, input);
            for p in input.get_ancestors(parents, *idx).iter(){
                if Some(*p)==break_point{
                    break;
                }
                for idx_ in self.place2idx[*idx].iter(){
                    let (s, d) = input.sd[*idx_];
                    should_opens[*p].insert(s);
                    should_opens[*p].insert(d);
                }
            }
        }
    }

    // 破壊的。 subtree以下破壊して再構築
    fn optimize_subtree(&mut self, input: &Input, subtree_root: usize, parents: &Vec<Option<usize>>, childrens: &Vec<Vec<usize>>){
        // まずまっさらにする
        let mut moves = vec![];
        for idx in input.get_subtree_idxs(childrens, subtree_root){
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        self.update(&NeighborParam{moves}, input);
        // subtreeより上のshould_openを求めておく。
        let mut should_opens: Vec<HashSet<usize>> = vec![HashSet::new(); input.h*input.w];
        for &idx in input.get_ancestors(parents, subtree_root).iter(){
            let (flgs_with_invalid, flgs) = self.get_reachmap_with_invalid(input, idx);
            let should_open = self.get_should_open(input, idx, flgs_with_invalid, flgs);
            for i in 0..(input.t*2){
                if should_open.is_ok(i){
                    should_opens[idx].insert(i);
                }
            }
        }
        // 葉の方から繰り返し。
        for idx in input.get_subtree_idxs(childrens, subtree_root).iter().rev(){
            // costsを求める。直近の親を重視しつつ、should_openを増やさないようにする。厳密にshould_openは求めず、treeのルートで行く前提。
            let n_ancestors = input.get_ancestors(parents, *idx).len();
            let base_cost = n_ancestors*(n_ancestors+1)/2;
            let mut costs = vec![base_cost; input.t*2];
            for (i, p) in input.get_ancestors(parents, *idx).iter().enumerate(){
                for j in should_opens[*p].iter(){
                    costs[*j] -= i+1;
                }
            }
            // neighbor取得してupdate
            let neighborparam = self.get_neighbor_optimize_single_place(input, *idx, &costs);
            self.update(&neighborparam, input);
            for p in input.get_ancestors(parents, *idx).iter(){
                for idx_ in self.place2idx[*idx].iter(){
                    let (s, d) = input.sd[*idx_];
                    should_opens[*p].insert(s);
                    should_opens[*p].insert(d);
                }
            }
        }
    }

    fn optimize_ancestors(&mut self, input: &Input, subtree_root: usize, parents: &Vec<Option<usize>>, childrens: &Vec<Vec<usize>>){
        // まずまっさらにする
        let mut moves = vec![];
        for idx in input.get_ancestors(parents, subtree_root){
            for i in self.place2idx[idx].iter(){
                moves.push((*i, idx, input.h*input.w));
            }
        }
        self.update(&NeighborParam{moves}, input);
        // subtreeより上のshould_openを求めておく。
        let mut should_opens: Vec<HashSet<usize>> = vec![HashSet::new(); input.h*input.w];
        for &idx in input.get_ancestors(parents, subtree_root).iter(){
            let (flgs_with_invalid, flgs) = self.get_reachmap_with_invalid(input, idx);
            let should_open = self.get_should_open(input, idx, flgs_with_invalid, flgs);
            for i in 0..(input.t*2){
                if should_open.is_ok(i){
                    should_opens[idx].insert(i);
                }
            }
        }
        for idx in input.get_ancestors(parents, subtree_root).iter(){
            // costsを求める。直近の親を重視しつつ、should_openを増やさないようにする。厳密にshould_openは求めず、treeのルートで行く前提。
            let n_ancestors = input.get_ancestors(parents, *idx).len();
            let base_cost = n_ancestors*(n_ancestors+1)/2;
            let mut costs = vec![base_cost; input.t*2];
            for (i, p) in input.get_ancestors(parents, *idx).iter().enumerate(){
                for j in should_opens[*p].iter(){
                    costs[*j] -= i+1;
                }
            }
            // neighbor取得してupdate
            let neighborparam = self.get_neighbor_optimize_single_place(input, *idx, &costs);
            self.update(&neighborparam, input);
            for p in input.get_ancestors(parents, *idx).iter(){
                for idx_ in self.place2idx[*idx].iter(){
                    let (s, d) = input.sd[*idx_];
                    should_opens[*p].insert(s);
                    should_opens[*p].insert(d);
                }
            }
        }
    }

    fn get_should_open_with_target(&self, input: &Input, idx: usize, flgs_with_invalid: Vec<Flg>, flgs: Vec<Flg>) -> Flg{
        let is_open = flgs[idx].clone();
        let mut should_open: Flg = Flg::zero();
        for (i, (flg_with_invalid, flg)) in flgs_with_invalid.iter().zip(flgs.iter()).enumerate(){
            // 元々is_okな前提なので、flg==0なら見る必要ない場所
            if flg==&Flg::zero(){
                continue;
            }
            let mut idxs = self.place2idx[i].iter().map(|&c| c).collect_vec();
            if input.art_points.contains(&i) && !input.art_point_children[i].contains(&idx){
                idxs = vec![];
                for i_ in input.art_point_children[i].iter(){
                    idxs.append(&mut self.place2idx[*i_].iter().map(|&c| c).collect_vec());
                }
            }
            for idx in idxs.iter(){
                let (s, d) = input.sd[*idx];
                let maxbit = flg.max_bit(s).unwrap();
                if let Some(maxbit_) = flg_with_invalid.max_bit(s){
                    if maxbit_ != maxbit{
                        should_open = should_open.or(&Flg::from_single(maxbit));
                    }
                } else {
                    should_open = should_open.or(&Flg::from_single(maxbit));
                }
                if !flg_with_invalid.is_ok(d){
                    should_open = should_open.or(&Flg::from_single(d));
                }
            }
        }
        should_open
    }

    fn get_should_open(&self, input: &Input, idx: usize, flgs_with_invalid: Vec<Flg>, flgs: Vec<Flg>) -> Flg{
        let is_open = flgs[idx].clone();
        let mut should_open: Flg = Flg::zero();
        for (i, (flg_with_invalid, flg)) in flgs_with_invalid.iter().zip(flgs.iter()).enumerate(){
            for idx in self.place2idx[i].iter(){
                let (s, d) = input.sd[*idx];
                let maxbit = flg.max_bit(s).unwrap();
                if let Some(maxbit_) = flg_with_invalid.max_bit(s){
                    if maxbit_ != maxbit{
                        should_open = should_open.or(&Flg::from_single(maxbit));
                    }
                } else {
                    should_open = should_open.or(&Flg::from_single(maxbit));
                }
                if !flg_with_invalid.is_ok(d){
                    should_open = should_open.or(&Flg::from_single(d));
                }
            }
        }
        should_open
    }

    fn get_neighbor_optimize_single_place_with_target(&mut self, input: &Input, idx: usize, costs: &Vec<usize>, from: (Flg, usize)) -> NeighborParam{
        // 元々is_okな前提
        let (flgs_with_invalid, flgs) = self.get_reachmap_with_invalid_with_target(input, idx, from.clone());
        // let (flgs_with_invalid, flgs) = self.get_reachmap_with_invalid_with_target(input, idx, tmp);
        // あけとくべきタイミングを求める
        let is_open = flgs[idx].clone();
        let should_open = self.get_should_open_with_target(input, idx, flgs_with_invalid, flgs);
        // 一旦消す。
        let before_idxs = self.place2idx[idx].iter().map(|&c| c).collect_vec();
        let tmp_neighbor = NeighborParam{moves: before_idxs.iter().map(|&i| (i, idx, input.h*input.w)).collect_vec()};
        self.update(&tmp_neighbor, input);
        // should_openを守りながら最大限詰め込む。同じスコアならば、作物少なくする
        let mut dp: Vec<(usize, Flg, Vec<usize>, usize)> = vec![(0, is_open, vec![], 0); input.t*2]; // (詰め込んだ時間, 詰め込んだi, 詰め込んだidxs, 詰め込んだコスト)
        for s in 0..(input.t*2-1){
            let (score, flg_, idxs, cost) = dp[s].clone();
            let mut flg = flg_;
            if should_open.is_ok(s){
                flg = flg.and(&Flg::from_invalid2(0, s));
            }
            // 作物埋める
            for &(d, i) in self.rest_sdi[s].iter(){
                if flg.is_ok(d){
                    if let Some(maxbit) = flg.max_bit(s){
                    // if flg.is_ok(s){
                        if should_open.and(&Flg::from_valid(maxbit, d))!=Flg::zero(){
                            continue;
                        }
                        let score_ = score + d-s+1;
                        let mut idxs_ = idxs.clone();
                        idxs_.push(i);
                        let ele = (score_, flg.and(&Flg::from_invalid(0, d)), idxs_, cost+costs[d]+costs[maxbit]);
                        if is_good_p2(&dp[d], &ele){
                            dp[d] = ele;
                        }
                    }
                }
            }
            // 　何もしない
            let ele = (score, flg, idxs, cost);
            if is_good_p2(&dp[s+1], &ele){
                dp[s+1] = ele;
            }
        }

        let mut moves = vec![];
        for i in before_idxs.iter(){
            moves.push((*i, idx, input.h*input.w));
        }
        for &i in dp[input.t*2-1].2.iter(){
            moves.push((i, input.h*input.w, idx));
        }

        self.undo(&tmp_neighbor, input);
        NeighborParam { moves }
    }


    fn get_neighbor_optimize_single_place(&mut self, input: &Input, idx: usize, costs: &Vec<usize>) -> NeighborParam{
        // 元々is_okな前提
        let (flgs_with_invalid, flgs) = self.get_reachmap_with_invalid(input, idx);
        // あけとくべきタイミングを求める
        let is_open = flgs[idx].clone();
        let should_open = self.get_should_open(input, idx, flgs_with_invalid, flgs);
        // 一旦消す。
        let before_idxs = self.place2idx[idx].iter().map(|&c| c).collect_vec();
        let tmp_neighbor = NeighborParam{moves: before_idxs.iter().map(|&i| (i, idx, input.h*input.w)).collect_vec()};
        self.update(&tmp_neighbor, input);
        // should_openを守りながら最大限詰め込む。同じスコアならば、作物少なくする
        let mut dp: Vec<(usize, Flg, Vec<usize>, usize)> = vec![(0, is_open, vec![], 0); input.t*2]; // (詰め込んだ時間, 詰め込んだi, 詰め込んだidxs, 詰め込んだコスト)
        for s in 0..(input.t*2-1){
            let (score, flg_, idxs, cost) = dp[s].clone();
            let mut flg = flg_;
            if should_open.is_ok(s){
                flg = flg.and(&Flg::from_invalid2(0, s));
            }
            // 作物埋める
            for &(d, i) in self.rest_sdi[s].iter(){
                if flg.is_ok(d){
                    if let Some(maxbit) = flg.max_bit(s){
                    // if flg.is_ok(s){
                        if should_open.and(&Flg::from_valid(maxbit, d))!=Flg::zero(){
                            continue;
                        }
                        let score_ = score + d-s+1;
                        let mut idxs_ = idxs.clone();
                        idxs_.push(i);
                        let ele = (score_, flg.and(&Flg::from_invalid(0, d)), idxs_, cost+costs[d]+costs[maxbit]);
                        if is_good_p2(&dp[d], &ele){
                            dp[d] = ele;
                        }
                    }
                }
            }
            // 　何もしない
            let ele = (score, flg, idxs, cost);
            if is_good_p2(&dp[s+1], &ele){
                dp[s+1] = ele;
            }
        }

        let mut moves = vec![];
        for i in before_idxs.iter(){
            moves.push((*i, idx, input.h*input.w));
        }
        for &i in dp[input.t*2-1].2.iter(){
            moves.push((i, input.h*input.w, idx));
        }

        self.undo(&tmp_neighbor, input);
        NeighborParam { moves }
    }

    fn update(&mut self, params: &NeighborParam, input: &Input){
        for (i, j, k) in params.moves.iter(){
            self.places[*i] = *k;
            self.place2idx[*j].remove(i);
            self.place2idx[*k].insert(*i);
            if *j==400{
                let (s, d) = input.sd[*i];
                self.rest_sdi[s].retain(|&(d_, i_)| i_!=*i);
            }
            if *k==400{
                let (s, d) = input.sd[*i];
                self.rest_sdi[s].push((d, *i));
                self.rest_sdi[s].sort_by_key(|&(d, _)| d);
            }
        }
    }

    fn undo(&mut self, params: &NeighborParam, input: &Input){
        for (i, j, k) in params.moves.iter(){
            self.places[*i] = *j;
            self.place2idx[*k].remove(i);
            self.place2idx[*j].insert(*i);
            if *k==400{
                let (s, d) = input.sd[*i];
                self.rest_sdi[s].retain(|&(d_, i_)| i_!=*i);
            }
            if *j==400{
                let (s, d) = input.sd[*i];
                self.rest_sdi[s].push((d, *i));
                self.rest_sdi[s].sort_by_key(|&(d, _)| d);
            }
        }
    }

    fn get_neighbor(&mut self, input: &Input) -> NeighborParam{
        let mut rng = rand::thread_rng();
        return self.get_neighbor_optimize_single_place(input, rng.gen::<usize>()%(input.h*input.w), &vec![0; input.t*2]);
        let mode_flg = rng.gen::<usize>()%100;
        let mut moves = vec![];
        if mode_flg<90{
            return self.get_neighbor_optimize_single_place(input, rng.gen::<usize>()%(input.h*input.w), &vec![0; input.t*2]);
        } else if mode_flg<91{
                //move
            for _ in 0..100{
                let i = **self.place2idx[input.h*input.w].iter().collect_vec().choose(&mut rng).unwrap();
                let k = rng.gen::<usize>()%(input.h*input.w);
                if (self.places[i]!=k){
                    moves.push((i, self.places[i], k));
                    break;
                }
            }
        } else if mode_flg<95 {
            //move
            for _ in 0..100{
                let i = rng.gen::<usize>()%input.k;
                let k = rng.gen::<usize>()%(input.h*input.w+1);
                if (self.places[i]!=k){
                    moves.push((i, self.places[i], rng.gen::<usize>()%(input.h*input.w)));
                    break;
                }
            }
        } else {
            // swap
            for _ in 0..100{
                let i1 = rng.gen::<usize>()%input.k;
                let i2 = rng.gen::<usize>()%input.k;
                let k1 = self.places[i1];
                let k2 = self.places[i2];
                if (k1!=k2){
                    moves.push((i1, k1, k2));
                    moves.push((i2, k2, k1));
                    break;
                }
            }
        }
        NeighborParam{moves}
    }

    fn get_score(&self, input: &Input, w: f64) -> f64{
        for (i, idxs_) in self.place2idx.iter().enumerate(){
            if i==input.h*input.w{
                continue;
            }
            let mut idxs = idxs_.iter().map(|&c| c).collect_vec();
            idxs.sort_by_key(|&i_| input.sd[i_].1);
            let mut flg = Flg::one();
            let mut last_d = 0;
            for idx in idxs.iter(){
                let (s, d) = input.sd[*idx];
                flg = flg.and(&input.flgs[*idx]);
                if let Some(s_) = flg.max_bit(s){
                    if s_<last_d{
                        return -10000000.0;
                    }
                    last_d = d;
                } else {
                    return -10000000.0;
                }
            }
        }

        let mut valid_x = 0;
        let mut invalid_x = 0;
        let mut map_x = 0;

        let mut flgs = self.get_reachmap(input);
        for (i, flg_) in flgs.iter().enumerate(){
            let mut idxs = self.place2idx[i].iter().map(|&c| c).collect_vec();
            let mut flg = flg_.clone();
            idxs.sort_by_key(|&i_| input.sd[i_].1);
            for idx in idxs.iter(){
                let (s, d) = input.sd[*idx];
                if !flg.is_ok(d){
                    return -10000000.0;
                }
                if let Some(s_) = flg.max_bit(s){
                    flg = flg.and(&Flg::from_invalid2(0, d));
                    valid_x += d-s+1;
                    invalid_x += s-s_;
                    map_x += (d-s_+1)*self.place2score[i];
                } else {
                    return -10000000.0;
                }
            }
        }
        (valid_x as f64)*3.0 - (invalid_x as f64)*10.0*w - (map_x as f64)*0.0
        // valid_x as f64
    }

    fn get_raw_score(&self, input: &Input) -> f64{
        if self.is_ok(input){
            let mut score = 0;
            for (i, place) in self.places.iter().enumerate(){
                if *place<input.h*input.w{
                    score += (input.sd[i].1-input.sd[i].0+1)/2;
                }
            }
            (score as f64)*1000000.0/((input.h*input.w*input.t) as f64)
        } else {
            -10000000.0
        }
    }

    fn get_raw_score_without_check(&self, input: &Input) -> f64{
        let mut score = 0;
        for (i, place) in self.places.iter().enumerate(){
            if *place<input.h*input.w{
                score += (input.sd[i].1-input.sd[i].0+1)/2;
            }
        }
        (score as f64)*1000000.0/((input.h*input.w*input.t) as f64)
    }

    fn get_single_score(&self, input: &Input, place: usize) -> f64{
        let mut score = 0;
        for &i in self.place2idx[place].iter(){
            score += (input.sd[i].1-input.sd[i].0+1)/2;
        }
        (score as f64)
    }

    fn print(&mut self, input: &Input){
        // is_ok前提
        let mut anss = vec![];
        let mut flgs = self.get_reachmap(input);
        for (i, flg) in flgs.iter().enumerate(){
            for idx in self.place2idx[i].iter(){
                let (s, d) = input.sd[*idx];
                let maxbit = flg.max_bit(s).unwrap();
                anss.push((idx+1, i/input.w, i%input.w ,maxbit/2+1));
            }
        }
        println!("{}", anss.len());
        for (k, i, j, s) in anss.iter(){
            println!("{} {} {} {}", k, i, j, s);
        }
    }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.get_raw_score_without_check(input);
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
        let mut new_state = state.clone();
        let (parents, childrens) = input.get_random_tree();

        let mut idx1 = rng.gen::<usize>()%(input.h*input.w);
        while (input.get_subtree_idxs(&childrens, idx1).len()<5 && input.get_subtree_idxs(&childrens, idx1).len()>=40){
            idx1 =  rng.gen::<usize>()%(input.h*input.w);
        }
        new_state.optimize_subtree_and_ancestors(input, idx1, &parents, &childrens);
        // new_state.optimize_art_point(input, idx1, &parents, &childrens);

        // let mut idx1 = rng.gen::<usize>()%input.h*input.w;
        // let mut idx2 = rng.gen::<usize>()%input.h*input.w;
        // while idx1==idx2 || (input.get_subtree_idxs(&childrens, idx1).len()>=20) || (input.get_subtree_idxs(&childrens, idx2).len()>=20){
        //     idx1 = rng.gen::<usize>()%input.h*input.w;
        //     idx2 = rng.gen::<usize>()%input.h*input.w;
        // }
        // new_state.optimize_subtrees(input, idx1, idx2, &parents, &childrens);

        // let mut idxs = HashSet::new();
        // while idxs.len()<5{
        //     idxs.insert(rng.gen::<usize>()%input.h*input.w);
        // }
        // for &idx in idxs.iter(){
        //     new_state.clear_place(input, idx);
        // }
        // for &idx in idxs.iter(){
        //     let neighbor = new_state.get_neighbor_optimize_single_place(input, idx, &vec![0; input.t*2]);
        //     new_state.update(&neighbor, input);
        // }

        let new_score = new_state.get_raw_score_without_check(input);
        // let new_score = cur_score;//new_state.get_raw_score_without_check(input);
        let score_diff = new_score-cur_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            eprintln!("{} {} {} {}", new_score, all_iter, new_state.get_single_score(input, idx1), input.get_subtree_idxs(&childrens, idx1).len());
            // eprintln!("{} {}", new_score, all_iter);
            accepted_cnt += 1;
            // eprintln!("{} {} {:?}", cur_score, new_score, all_iter);
            cur_score = new_score;
            state = new_state;
            last_updated = all_iter;
            //state.print(input);
            if new_score>best_score{
                best_state = state.clone();
                best_score = new_score;
            }
        } else {
            //state.undo(&neighbor, input);
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("score  : {}", best_state.get_raw_score_without_check(input));
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
    // eprintln!("{}", init_state.get_score(input));
    //init_state.print(input);
    // eprintln!("{:?}", init_state.get_mapflg(input, input.i0*input.w));
    // eprintln!("{:?}", init_state.get_reachmap(input)[input.i0*input.w]);
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
    // eprintln!("{}", best_state.is_ok(input));
    // let reachmap = best_state.get_reachmap(input);
    // eprintln!("{:b} {:b}", reachmap[input.i0*input.w].bit0, reachmap[input.i0*input.w].bit1);
    // // for t in 0..200{
    //     let mut tmp = vec![vec![]; 20];
    //     for i in 0..20{
    //         for j in 0..20{
    //             tmp[i].push(reachmap[i*input.w+j].is_ok(t));
    //         }
    //     }
    // }
}
