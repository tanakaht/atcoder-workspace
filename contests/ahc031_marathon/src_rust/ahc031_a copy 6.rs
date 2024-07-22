#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
use std::vec;
use itertools::Itertools;
use num::{range, ToPrimitive};
use num_integer::Roots;
use proconio::source::line;
use proconio::{input, marker::Chars};
use rand_core::{block, le};
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
const W: usize = 1000;

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    D: usize,
    N: usize,
    A: Vec<Vec<usize>>,
    rest_rates: Vec<usize>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            _: usize,
            D: usize,
            N: usize,
            A: [[usize; N]; D], // a is Vec<i32>, n-array.
            //m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let mut rest_rates: Vec<usize> = vec![];
        for d in 0..D {
            rest_rates.push((W * W - A[d].iter().sum::<usize>()) / N);
        }
        Self { D, N, A, rest_rates }
    }

    fn get_score(&self, ans: &HashMap<(usize, usize), (usize, usize, usize, usize)>) -> i64 {
        let mut score = 1;
        for d in 0..self.D {
            for n in 0..self.N {
                let (x1, y1, x2, y2) = *ans.get(&(d, n)).unwrap();
                let space = (x2 - x1) * (y2 - y1);
                score += ((self.A[d][n]-std::cmp::min(space, self.A[d][n]))*100) as i64;
            }
        }
        let mut w_lines = vec![Line::new();W+1];
        let mut h_lines = vec![Line::new();W+1];
        w_lines[0].add(0, W);
        w_lines[W].add(0, W);
        h_lines[0].add(0, W);
        h_lines[W].add(0, W);
        for n in 0..self.N{
            let (x1, y1, x2, y2) = *ans.get(&(0, n)).unwrap();
            w_lines[x1].add(y1, y2);
            w_lines[x2].add(y1, y2);
            h_lines[y1].add(x1, x2);
            h_lines[y2].add(x1, x2);
        }
        for d in 1..self.D{
            let mut new_w_lines = vec![Line::new();W+1];
            let mut new_h_lines = vec![Line::new();W+1];
            new_w_lines[0].add(0, W);
            new_w_lines[W].add(0, W);
            new_h_lines[0].add(0, W);
            new_h_lines[W].add(0, W);
            for n in 0..self.N{
                let (x1, y1, x2, y2) = *ans.get(&(d, n)).unwrap();
                new_w_lines[x1].add(y1, y2);
                new_w_lines[x2].add(y1, y2);
                new_h_lines[y1].add(x1, x2);
                new_h_lines[y2].add(x1, x2);
            }
            for i in 0..W+1{
                score += w_lines[i].diff(&new_w_lines[i]) as i64;
                score += h_lines[i].diff(&new_h_lines[i]) as i64;
                // eprintln!("w:{} {:?} {:?}", w_lines[i].diff(&new_w_lines[i]), w_lines[i].availables, new_w_lines[i].availables);
            }
            w_lines = new_w_lines;
            h_lines = new_h_lines;
            // eprintln!("score at d{}: {}", d, score);
        }
        score
    }
}

#[derive(Debug, Clone)]
struct Line{
    availables: Vec<(usize, usize)>,
}

impl Line{
    fn new() -> Self{
        let mut availables = vec![];
        Self{availables}
    }

    fn diff(&self, other: &Line)-> usize{
        let mut events = vec![];
        for &(fr, to) in self.availables.iter(){
            events.push((fr, 1, 0));
            events.push((to, -1, 0));
        }
        for &(fr, to) in other.availables.iter(){
            events.push((fr, 0, 1));
            events.push((to, 0, -1));
        }
        events.sort();
        let mut cur = (0, 0, 0);
        let mut score = 0;
        for (idx, flg1, flg2) in events{
            cur = (cur.0, cur.1+flg1, cur.2+flg2);
            if cur.1!=cur.2{
                cur = (idx, cur.1, cur.2);
            } else {
                score += idx-cur.0;
                cur = (idx, cur.1, cur.2);
            }
        }
        score
    }

    fn add(&mut self, x: usize, y: usize){
        let mut new_available = vec![];
        let mut events = vec![(x, 1), (y, -1)];
        for &(fr, to) in self.availables.iter(){
            events.push((fr, 1));
            events.push((to, -1));
        }
        events.sort();
        let mut cur = (0, 0);
        for (idx, flg) in events{
            cur = (cur.0, cur.1+flg);
            if cur.1==0{
                new_available.push((cur.0, idx));
            }
            if flg==1 && cur.1==1{
                cur = (idx, 1);
            }
        }
        self.availables = new_available;
    }
}

struct Neighbor{
    x: usize
}

struct GroupTree{
    groups: Vec<Group>,
    daily_nodes: Vec<HashSet<usize>>,
    startdaywise_nodes: Vec<HashSet<usize>>,
    t_nodes: HashSet<usize>,
}

impl GroupTree{
    fn new(input: &Input) -> Self{
        let mut groups = vec![];
        let mut daily_nodes = vec![HashSet::new(); input.D];
        let mut startdaywise_nodes = vec![HashSet::new(); input.D];
        for d in 0..input.D{
            for n in 0..input.N{
                groups.push(Group::new(input,  d, d+1, vec![vec![n]]));
                daily_nodes[d].insert(d*input.N+n);
                startdaywise_nodes[d].insert(d*input.N+n);
            }
        }
        let mut daily_roots = daily_nodes.clone();
        let mut ret = Self{groups, daily_nodes, startdaywise_nodes, t_nodes: HashSet::new()};
        let mut rng = thread_rng();
        let mut joint_t_idxs = vec![];
        for d in 0..input.D{
            while daily_roots[d].len()>1{
                let cloned = daily_roots[d].iter().cloned().collect::<Vec<_>>();
                let idxs = cloned.iter().choose_multiple(&mut rng, 2);
                ret.joint_a(input, *idxs[0], *idxs[1]);
                daily_roots[d].remove(idxs[0]);
                daily_roots[d].remove(idxs[1]);
                daily_roots[d].insert(ret.groups.len()-1);
            }
            joint_t_idxs.push(*daily_roots[d].iter().next().unwrap());
        }
        ret.joint_t(input, &joint_t_idxs);
        ret
    }

    fn get_root_idx(&self) -> usize{
        let mut ret = 0;
        while let Some(parent_idx) = self.groups[ret].parent_group_idx{
            ret = parent_idx;
        }
        ret
    }

    fn get_t_parent(&self, idx: usize) -> Option<usize>{
        if let Some(mut ret) = self.groups[idx].parent_group_idx{
            while let Some(parent_idx) = self.groups[ret].parent_group_idx{
                if self.groups[parent_idx].group_type=="t"{
                    return Some(parent_idx);
                }
                ret = parent_idx;
            }
            None
        } else {
            None
        }
    }

    fn abridge_t(&mut self, input: &Input, child_idx: usize){
        let parent_idx = self.groups[child_idx].parent_group_idx.unwrap();
        let child_group_idx = self.groups[child_idx].child_group_idx.clone();
        let mut new_idxs = self.groups[parent_idx].child_group_idx.clone();
        new_idxs.retain(|&x| x!=child_idx);
        for &mago_idx in child_group_idx.iter(){
            self.groups[mago_idx].parent_group_idx = Some(parent_idx);
            new_idxs.push(mago_idx);
        }
        new_idxs.sort_by_key(|x| self.groups[*x].start_day);
        self.groups[parent_idx].child_group_idx = new_idxs;
    }

    fn remove_child_from_a(&mut self, input: &Input, child_idx: usize) -> usize{
        // a_node前提
        let parent_idx = self.groups[child_idx].parent_group_idx.unwrap();
        assert!(self.groups[parent_idx].group_type=="a");
        let other_child_idx = if self.groups[parent_idx].child_group_idx[0]==child_idx{
            self.groups[parent_idx].child_group_idx[1]
        } else {
            self.groups[parent_idx].child_group_idx[0]
        };
        if let Some(ascent_idx) = self.groups[parent_idx].parent_group_idx{
            self.groups[ascent_idx].child_group_idx = self.groups[ascent_idx].child_group_idx.iter().map(|&x| if x==parent_idx{other_child_idx}else{x}).collect();
            self.groups[other_child_idx].parent_group_idx = Some(ascent_idx);
            self.refresh(input, other_child_idx);
            if self.groups[other_child_idx].group_type=="t"&&self.groups[ascent_idx].group_type=="t"{
                self.abridge_t(input, other_child_idx);
            }
        }
        self.remove_node(parent_idx);
        other_child_idx
    }

    fn split_t(&mut self, input: &Input, target_idx: usize, start_day: usize, end_day: usize)->usize{
        if self.groups[target_idx].start_day==start_day && self.groups[target_idx].end_day==end_day{
            return target_idx;
        }
        assert!(self.groups[target_idx].group_type=="t");
        assert!(self.groups[target_idx].start_day<=start_day && self.groups[target_idx].end_day>=end_day);
        let mut idxs = self.groups[target_idx].child_group_idx.clone();
        idxs.retain(|&x| self.groups[x].start_day>=start_day && self.groups[x].end_day<=end_day);
        if idxs.len()==1{
            assert!(self.groups[idxs[0]].start_day==start_day && self.groups[idxs[0]].end_day==end_day);
            return idxs[0];
        }
        let new_group_idx = self.joint_t(input, &idxs);
        assert!(self.groups[new_group_idx].start_day==start_day && self.groups[new_group_idx].end_day==end_day);
        self.groups[new_group_idx].parent_group_idx = Some(target_idx);
        let mut new_idxs = self.groups[target_idx].child_group_idx.clone();
        new_idxs.retain(|&x| !idxs.contains(&x));
        new_idxs.push(new_group_idx);
        new_idxs.sort_by_key(|x| self.groups[*x].start_day);
        self.groups[target_idx].child_group_idx = new_idxs;
        new_group_idx
    }

    fn insert_a(&mut self, input: &Input, idx: usize, target_idx: usize)->usize{
        let target_idx = self.split_t(input, target_idx, self.groups[idx].start_day, self.groups[idx].end_day);
        let ascent_idx = self.groups[target_idx].parent_group_idx;
        let new_group_idx = self.joint_a(input, idx, target_idx);
        self.groups[new_group_idx].parent_group_idx = ascent_idx;
        if let Some(ascent_idx) = ascent_idx{
            self.groups[ascent_idx].child_group_idx = self.groups[ascent_idx].child_group_idx.iter().map(|&x| if x==target_idx{new_group_idx}else{x}).collect();
            self.refresh(input, new_group_idx);
        }
        new_group_idx
    }

    fn forced_joint_t(&mut self, input: &Input, idxs: &Vec<usize>) -> Vec<(usize, usize, usize)>{
        let target_t_node = self.get_t_parent(idxs[0]).unwrap();
        for idx in idxs[1..].iter(){
            assert!(self.get_t_parent(*idx)==Some(target_t_node));
        }
        let mut undo_params = vec![];
        // 直属の親からひっぺがす
        for &idx in idxs{
            undo_params.push((0, idx, self.remove_child_from_a(input, idx)));
        }
        // joint_tした新しいgroupを作る
        let new_group_idx = self.joint_t(input, idxs);
        // target_t_nodeとjoint_aする
        self.insert_a(input, new_group_idx, target_t_node);
        undo_params.push((1, new_group_idx, 0));
        undo_params.reverse();
        undo_params
    }

    fn undo_forced_joint_t(&mut self, input: &Input, undo_params: &Vec<(usize, usize, usize)>){
        for &(flg, idx1, idx2) in undo_params.iter(){
            if flg==0{
                self.insert_a(input, idx1, idx2);
            } else if flg==1{
                self.remove_child_from_a(input, idx1);
            } else{
                panic!("unknow flg");
            }
        }
    }

    fn get_best_children_idxs(&self, idx: usize, target_v: usize, n: usize) -> Option<Vec<usize>>{
        let mut ret = vec![];
        let mut best: HashMap<usize, (usize, usize)> = HashMap::new();
        let mut q = vec![];
        for &child_idx in self.groups[idx].child_group_idx.iter(){
            for &mago_idx in self.groups[child_idx].child_group_idx.iter(){
                q.push(mago_idx);
            }
        }
        while !q.is_empty(){
            let idx = q.pop().unwrap();
            let g = &self.groups[idx];
            for &idx_ in g.child_group_idx.iter(){
                q.push(idx_);
            }
            let max_area = *g.member_sums.iter().max().unwrap();
            if max_area>target_v || g.member_idxs[0].len()!=n{
                continue;
            }
            if let Some(&(v, _)) = best.get(&g.start_day){
                if v>(target_v-max_area){
                    best.insert(g.start_day, (target_v-max_area, idx));
                }
            } else {
                best.insert(g.start_day, (target_v-max_area, idx));
            }
        }
        let mut cur = self.groups[idx].start_day;
        for (k, (_, idx)) in best.iter().sorted_by_key(|(x, _)| **x){
            ret.push(*idx);
            if self.groups[*idx].start_day!=cur{
                return None;
            }
            cur = self.groups[*idx].end_day;
        }
        if cur!=self.groups[idx].end_day{
            return None;
        }
        Some(ret)
    }

    fn get_children_idxs(&self, idx: usize) -> HashMap<(usize, usize), Vec<usize>>{
        let mut ret = HashMap::new();
        let g = &self.groups[idx];
        ret.insert((g.start_day, g.member_idxs[0].len()), vec![idx]);
        for child_idx in self.groups[idx].child_group_idx.iter(){
            if self.groups[*child_idx].group_type=="t"{
                let g_ = &self.groups[*child_idx];
                if ret.contains_key(&(g_.start_day, g_.member_idxs[0].len())){
                    ret.get_mut(&(g_.start_day, g_.member_idxs[0].len())).unwrap().push(*child_idx);
                } else {
                    ret.insert((g_.start_day, g_.member_idxs[0].len()), vec![*child_idx]);
                }
            } else {
                for (k, v) in self.get_children_idxs(*child_idx){
                    if ret.contains_key(&k){
                        ret.get_mut(&k).unwrap().append(&mut v.clone());
                    } else {
                        ret.insert(k, v);
                    }

                }
            }
        }
        ret
    }

    // 0: だめ, 1: joint_a, 2: joint_t
    fn can_joint(&self, idx1: usize, idx2: usize) -> usize{
        if self.groups[idx1].start_day==self.groups[idx2].start_day && self.groups[idx1].end_day==self.groups[idx2].end_day{
            1
        } else if self.groups[idx1].end_day==self.groups[idx2].start_day && self.groups[idx1].member_idxs[0].len()==self.groups[idx2].member_idxs[0].len(){
            2
        } else {
            0
        }
    }

    fn refresh(&mut self, input: &Input, idx: usize){
        if let Some(parent_idx) = self.groups[idx].parent_group_idx{
            if self.groups[parent_idx].group_type=="t"{
                let child_idx = self.groups[parent_idx].child_group_idx.clone();
                let ascent_idx = self.groups[parent_idx].parent_group_idx;
                if let Some(new_group) = Group::joint_t(input, &child_idx.iter().map(|i| &self.groups[*i]).collect()){
                    self.groups[parent_idx] = new_group;
                    self.groups[parent_idx].child_group_idx = child_idx;
                    self.groups[parent_idx].parent_group_idx = ascent_idx;
                    self.refresh(input, parent_idx);
                }
            } else {
                let child_idx = self.groups[parent_idx].child_group_idx.clone();
                let ascent_idx = self.groups[parent_idx].parent_group_idx;
                if let Some(new_group) = Group::joint_a(input, &self.groups[child_idx[0]], &self.groups[child_idx[1]]){
                    self.groups[parent_idx] = new_group;
                    self.groups[parent_idx].child_group_idx = child_idx;
                    self.groups[parent_idx].parent_group_idx = ascent_idx;
                    self.refresh(input, parent_idx);
                }
            }
        }
    }

    // fn random_joint(&mut self, input: &Input){
    //     // 適当に一つ選んでjoin
    //     let mut rng = thread_rng();
    //     let mut roots = self.roots.iter().cloned().collect::<Vec<_>>();
    //     roots.shuffle(&mut rng);
    //     for idx1 in roots{
    //         for &idx2 in self.roots.iter() {
    //             if idx1 == idx2 {
    //                 continue;
    //             }
    //             let can_joint = self.can_joint(idx1, idx2);
    //             if can_joint == 1 {
    //                 self.joint_a(input, idx1, idx2);
    //                 break;
    //             } else if can_joint == 2 {
    //                 self.joint_t(input, &vec![idx1, idx2]);
    //                 break;
    //             }
    //         }
    //     }
    // }

    fn remove_node(&mut self, idx: usize){
        let g = &self.groups[idx];
        for d in g.start_day..g.end_day{
            self.daily_nodes[d].remove(&idx);
        }
        self.startdaywise_nodes[g.start_day].remove(&idx);
        if g.group_type=="t"{
            self.t_nodes.remove(&idx);
        }
        if self.groups.len()-1==idx{
            self.groups.pop();
        } else {
            self.groups[idx] = Group::empty();
        }
    }

    fn add_node(&mut self, idx: usize){
        let g = &self.groups[idx];
        // daily_nodes: Vec<HashSet<usize>>,
        // startdaywise_nodes: Vec<HashSet<usize>>,
        // t_nodes: HashSet<usize>,
        for d in g.start_day..g.end_day{
            self.daily_nodes[d].insert(idx);
        }
        self.startdaywise_nodes[g.start_day].insert(idx);
        if g.group_type=="t"{
            self.t_nodes.insert(idx);
        }
    }

    // fn unjoint(&mut self, input: &Input, idx: usize){
    //     let idx1 = self.groups[idx].child_group_idx[0];
    //     let idx2 = self.groups[idx].child_group_idx[1];
    //     self.groups[idx1].parent_group_idx = None;
    //     self.groups[idx2].parent_group_idx = None;
    //     self.add_root(idx1);
    //     self.add_root(idx2);
    //     self.remove_root(idx);
    //     if self.groups.len()==idx+1{
    //         self.groups.pop();
    //     }
    // }

    fn joint_a(&mut self, input: &Input, idx1: usize, idx2: usize)->usize{
        assert!(self.groups[idx1].start_day==self.groups[idx2].start_day);
        assert!(self.groups[idx1].end_day==self.groups[idx2].end_day);
        let mut new_group = Group::joint_a(input, &self.groups[idx1], &self.groups[idx2]).unwrap();
        let new_group_idx = self.groups.len();
        self.groups[idx1].parent_group_idx = Some(new_group_idx);
        self.groups[idx2].parent_group_idx = Some(new_group_idx);
        new_group.child_group_idx.push(idx1);
        new_group.child_group_idx.push(idx2);
        self.groups.push(new_group);
        self.add_node(new_group_idx);
        new_group_idx
    }

    fn print_tree(&self){
        let mut node_type = vec![];
        let mut parents = vec![];
        for g in self.groups.iter(){
            node_type.push(g.group_type.clone());
            if let Some(parent_idx) = g.parent_group_idx{
                parents.push(parent_idx+1)
            } else{
                parents.push(0);
            }
        }
        eprintln!("{:?}", node_type);
        eprintln!("{:?}", parents);
    }

    fn joint_t(&mut self, input: &Input, idxs: &Vec<usize>)->usize{
        let mut new_group = Group::joint_t(input, &idxs.iter().map(|i| &self.groups[*i]).collect()).unwrap();
        let new_group_idx = self.groups.len();
        for idx in idxs{
            self.groups[*idx].parent_group_idx = Some(new_group_idx);
            new_group.child_group_idx.push(*idx);
        }
        self.groups.push(new_group);
        self.add_node(new_group_idx);
        new_group_idx
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone, Hash)]
struct Group{
    start_day: usize,
    end_day: usize,
    member_idxs: Vec<Vec<usize>>,
    member_sums: Vec<usize>,
    space: usize,
    line_needed: [usize; 1001],
    child_group_idx: Vec<usize>,
    parent_group_idx: Option<usize>,
    score: i64,
    group_type: String,
}

impl Group{
    fn new(input: &Input, start_day: usize, end_day: usize, member_idxs: Vec<Vec<usize>>) -> Self{
        let mut member_sums = vec![];
        for d in start_day..end_day{
            member_sums.push(member_idxs[d-start_day].iter().map(|idx| input.A[d][*idx]).sum());
        }
        let space = *member_sums.iter().max().unwrap();
        let mut line_needed: [usize; 1001] = [0; 1001];
        for l in 1..(W+1){
            line_needed[l] = (space+l-1)/l;
        }
        Self{start_day, end_day, member_idxs, member_sums, space, line_needed, child_group_idx: vec![], parent_group_idx: None, score: i64::MAX, group_type: "l".to_string()}
    }

    fn printdiff(&self, other: &Group){
        if self.start_day!=other.start_day{
            eprintln!("start_day: {}->{}", self.start_day, other.start_day);
        }
        if self.end_day!=other.end_day{
            eprintln!("end_day: {}->{}", self.end_day, other.end_day);
        }
        if self.member_idxs!=other.member_idxs{
            eprintln!("member_idxs: {:?}->{:?}", self.member_idxs, other.member_idxs);
        }

        if self.child_group_idx!=other.child_group_idx{
            eprintln!("child_group_idx: {:?}->{:?}", self.child_group_idx, other.child_group_idx);
        }
        if self.parent_group_idx!=other.parent_group_idx{
            eprintln!("parent_group_idx: {:?}->{:?}", self.parent_group_idx, other.parent_group_idx);
        }
        // start_day: usize,
        // end_day: usize,
        // member_idxs: Vec<Vec<usize>>,
        // member_sums: Vec<usize>,
        // space: usize,
        // line_needed: [usize; 1001],
        // child_group_idx: Vec<usize>,
        // parent_group_idx: Option<usize>,
        // score: i64,
        // group_type: String,

    }

    fn empty() -> Self{
        Self{start_day: 0, end_day: 0, member_idxs: vec![], member_sums: vec![], space: 0, line_needed: [0; 1001], child_group_idx: vec![], parent_group_idx: None, score: i64::MAX, group_type: "n".to_string()}
    }

    fn update_line_needed(&mut self, input: &Input, group_tree: &GroupTree){
        if self.group_type=="l"{
            return;
        } else if self.group_type=="t"{
            let mut line_needed: [usize; 1001] = [0; 1001];
            for idx in self.child_group_idx.iter(){
                let g = &group_tree.groups[*idx];
                for l in 1..(W+1){
                    line_needed[l] = std::cmp::max(line_needed[l], g.line_needed[l]);
                }
            }
            self.line_needed = line_needed;
        } else if self.group_type=="a"{
            let g1 = &group_tree.groups[self.child_group_idx[0]];
            let g2 = &group_tree.groups[self.child_group_idx[1]];
            let mut line_needed: [usize; 1001] = [0; 1001];
            for l in 1..(W+1){
                line_needed[l] = g1.line_needed[l] + g2.line_needed[l];
            }
            for l in 1..(W+1){
                let l1 = g1.line_needed[l];
                let l2 = g2.line_needed[l];
                if l1+l2<=1000 && line_needed[l1+l2]>l{
                    line_needed[l1+l2] = l;
                }
            }
            self.line_needed = line_needed;
        }
    }

    fn joint_t(input: &Input, gs: &Vec<&Group>) -> Option<Group>{
        for i in 0..gs.len()-1{
            if gs[i].end_day !=gs[i+1].start_day || gs[i].member_idxs[0].len()!=gs[i+1].member_idxs[0].len(){
                return None
            }
        }
        let mut new_member_idxs = vec![vec![]; gs.last().unwrap().end_day-gs[0].start_day];
        for g in gs.iter(){
            for d in 0..g.member_idxs.len(){
                for &i in g.member_idxs[d].iter(){
                    new_member_idxs[d+g.start_day-gs[0].start_day].push(i);
                }
            }
        }
        let mut member_sums = vec![];
        for d in gs[0].start_day..gs.last().unwrap().end_day{
            member_sums.push(new_member_idxs[d-gs[0].start_day].iter().map(|idx| input.A[d][*idx]).sum());
        }
        let space = *member_sums.iter().max().unwrap();
        let mut line_needed: [usize; 1001] = [0; 1001];
        for g in gs.iter(){
            for l in 1..(W+1){
                line_needed[l] = std::cmp::max(line_needed[l], g.line_needed[l]);
            }
        }
        Some(Self{start_day: gs[0].start_day, end_day: gs.last().unwrap().end_day, member_idxs: new_member_idxs, member_sums, space, line_needed, child_group_idx: vec![], parent_group_idx: None, score: i64::MAX, group_type: "t".to_string()})
    }

    fn joint_a(input: &Input, g1: &Group, g2: &Group) -> Option<Group>{
        if g1.start_day!=g2.start_day || g1.end_day!=g2.end_day{
            return None
        }
        let mut new_member_idxs = vec![vec![]; g1.member_idxs.len()];
        for d in 0..g1.member_idxs.len(){
            for &i in g1.member_idxs[d].iter(){
                new_member_idxs[d].push(i);
            }
            for &i in g2.member_idxs[d].iter(){
                new_member_idxs[d].push(i);
            }
        }
        let mut member_sums = vec![];
        for d in g1.start_day..g1.end_day{
            member_sums.push(new_member_idxs[d-g1.start_day].iter().map(|idx| input.A[d][*idx]).sum());
        }
        let space = *member_sums.iter().max().unwrap();
        let mut line_needed: [usize; 1001] = [0; 1001];
        for l in 1..(W+1){
            line_needed[l] = g1.line_needed[l] + g2.line_needed[l];
        }
        for l in 1..(W+1){
            let l1 = g1.line_needed[l];
            let l2 = g2.line_needed[l];
            if l1+l2<=1000 && line_needed[l1+l2]>l{
                line_needed[l1+l2] = l;
            }
        }
        for l in 1..(W){
            line_needed[l+1] = std::cmp::min(line_needed[l], line_needed[l+1]);
        }
        Some(Self{start_day: g1.start_day, end_day: g1.end_day, member_idxs: new_member_idxs, member_sums, space, line_needed, child_group_idx: vec![], parent_group_idx: None, score: i64::MAX, group_type: "a".to_string()})
    }

    fn solve(&self, input: &Input, rect: (usize, usize, usize, usize), group_tree: &GroupTree) -> HashMap<(usize, usize), (usize, usize, usize, usize)>{
        let mut ans = HashMap::new();
        if self.member_idxs[0].len() == 1{
            for d in self.start_day..self.end_day{
                ans.insert((d, self.member_idxs[d-self.start_day][0]), rect);
            }
            ans
        } else if self.group_type=="t"{
            for idx in self.child_group_idx.iter(){
                for (k, v) in group_tree.groups[*idx].solve(input, rect, group_tree).iter(){
                    ans.insert(*k, *v);
                }
            }
            ans
        } else {
            let group1_idx;
            let group2_idx;
            if group_tree.groups[self.child_group_idx[0]].member_idxs[0].len() <= group_tree.groups[self.child_group_idx[1]].member_idxs[0].len(){
                group1_idx = self.child_group_idx[0];
                group2_idx = self.child_group_idx[1];
            } else {
                group1_idx = self.child_group_idx[1];
                group2_idx = self.child_group_idx[0];
            }
            let ans1;
            let ans2;
            let (w, h) = (rect.2-rect.0, rect.3-rect.1);
            // let total_space1 = group_tree.groups[group1_idx].space+std::cmp::min(group_tree.groups[group1_idx].member_idxs[0].len()*std::cmp::min(rect.2-rect.0, rect.3-rect.1), group_tree.groups[group1_idx].blank_space);
            // let total_space2 = group_tree.groups[group2_idx].space+std::cmp::min(group_tree.groups[group2_idx].member_idxs[0].len()*std::cmp::min(rect.2-rect.0, rect.3-rect.1), group_tree.groups[group2_idx].blank_space);
            if group_tree.groups[group1_idx].line_needed[h]+group_tree.groups[group2_idx].line_needed[h]<=w{
                let w1 = group_tree.groups[group1_idx].line_needed[h];
                let w2 = group_tree.groups[group2_idx].line_needed[h];
                ans2 = group_tree.groups[group2_idx].solve(input, (rect.0+w1, rect.1, rect.0+w1+w2, rect.3), group_tree);
                ans1 = group_tree.groups[group1_idx].solve(input, (rect.0, rect.1, rect.0+w1, rect.3), group_tree);
                // ans2 = group_tree.groups[group2_idx].solve(input, (rect.0+w1, rect.1, rect.2, rect.3), group_tree);
            } else if group_tree.groups[group1_idx].line_needed[w]+group_tree.groups[group2_idx].line_needed[w]<=h{
                let h1 = group_tree.groups[group1_idx].line_needed[w];
                let h2 = group_tree.groups[group2_idx].line_needed[w];
                ans2 = group_tree.groups[group2_idx].solve(input, (rect.0, rect.1+h1, rect.2, rect.1+h1+h2), group_tree);
                ans1 = group_tree.groups[group1_idx].solve(input, (rect.0, rect.1, rect.2, rect.1+h1), group_tree);
                // ans2 = group_tree.groups[group2_idx].solve(input, (rect.0, rect.1+h1, rect.2, rect.3), group_tree);
            } else if w<h {
                let h1 = std::cmp::min(group_tree.groups[group1_idx].line_needed[w]-1, h-group_tree.groups[group2_idx].member_idxs[0].len());
                ans1 = group_tree.groups[group1_idx].solve(input, (rect.0, rect.1, rect.2, rect.1+h1), group_tree);
                ans2 = group_tree.groups[group2_idx].solve(input, (rect.0, rect.1+h1, rect.2, rect.3), group_tree);
            } else {
                let w1 = std::cmp::min(group_tree.groups[group1_idx].line_needed[h]-1, w-group_tree.groups[group2_idx].member_idxs[0].len());
                ans1 = group_tree.groups[group1_idx].solve(input, (rect.0, rect.1, rect.0+w1, rect.3), group_tree);
                ans2 = group_tree.groups[group2_idx].solve(input, (rect.0+w1, rect.1, rect.2, rect.3), group_tree);
            }
            for (k, v) in ans1.iter(){
                ans.insert(*k, *v);
            }
            for (k, v) in ans2.iter(){
                ans.insert(*k, *v);
            }
            ans
        }
    }

    fn get_score(&mut self) -> i64{
        if self.score != i64::MAX{
            return self.score;
        }
        let mut score = 0;
        // TODO: score計算
        self.score = score;
        score
    }
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    let mut best_ans = HashMap::new();
    let mut best_score = i64::MAX;
    while timer.elapsed().as_secs_f64() < 1.8 {
        let ans = solve(&input, &timer, 1.8);
        let score = input.get_score(&ans);
        eprintln!("score: {}", score);
        if score < best_score {
            best_score = score;
            best_ans = ans;
        }
    }
    for d in 0..input.D{
        for n in 0..input.N{
            if !best_ans.contains_key(&(d, n)){
                eprintln!("{} {}", d, n);
            }
            let (x1, y1, x2, y2) = best_ans[&(d, n)];
            println!("{} {} {} {}", x1, y1, x2, y2);
        }
    }
    eprintln!("score: {}", input.get_score(&best_ans));
}

fn solve(input: &Input, timer:&Instant, tl: f64) -> HashMap<(usize, usize), (usize, usize, usize, usize)>{
    let mut group_tree = GroupTree::new(input);
    let mut rng = thread_rng();
    // 時間方向にjoinできる限りしちゃう
    // while group_tree.roots.len()>1{
    //     group_tree.random_joint(input);
    //     let mut infos = vec![];
    //     for d in 0..input.D{
    //         for idx in group_tree.startdaywise_roots[d].iter(){
    //             let group = &group_tree.groups[*idx];
    //             infos.push((group.start_day, group.end_day, group.member_idxs[0].len()));
    //         }
    //     }
    //     eprintln!("{:?}", infos);
    // }
    // 残りを適当にjoin
    // for d in 0..input.D{
    //     while group_tree.daily_roots[d].len()>1{
    //         let idxs = group_tree.daily_roots[d].iter().choose_multiple(&mut rng, 2);
    //         group_tree.joint_a(input, *idxs[0], *idxs[1]);
    //     }
    // }
    // let mut idxs = vec![];
    // for d in 0..input.D{
    //     idxs.push(*group_tree.startdaywise_roots[d].iter().next().unwrap());
    // }
    // group_tree.joint_t(input, &idxs);
    // for d in 1..input.D{
    //     let idx1 = *group_tree.startdaywise_roots[0].iter().next().unwrap();
    //     let idx2 = *group_tree.startdaywise_roots[d].iter().next().unwrap();
    //     // eprintln!("0: {} {}",  group_tree.groups[idx1].space, group_tree.groups[idx1].blank_space);
    //     // eprintln!("{}: {} {}",  d, group_tree.groups[idx2].space, group_tree.groups[idx2].blank_space);
    //     group_tree.joint_t(input, &vec![idx1, idx2]);
    // }
    // eprintln!("{:?} {}", group_tree.groups[*group_tree.roots.iter().next().unwrap()].space, group_tree.groups[*group_tree.roots.iter().next().unwrap()].blank_space);
    // group_tree.print_tree();
    let target_idx = group_tree.groups.len()-1;
    // while timer.elapsed().as_secs_f64()<tl{
    for _ in 0..1000{
        let idx1 = rng.gen::<usize>()%input.N*(input.D-1);
        if group_tree.get_t_parent(idx1)!=Some(target_idx){
            continue;
        }
        if let Some(idxs) = group_tree.get_best_children_idxs(target_idx, group_tree.groups[idx1].space/W*1000, 1){
            let undo_params = group_tree.forced_joint_t(input, &idxs);
            if group_tree.groups[group_tree.get_root_idx()].line_needed[W]>W{
                group_tree.undo_forced_joint_t(input, &undo_params);
            } else {
                eprintln!("join: {:?}", idxs);
            }
        }
    }
    while timer.elapsed().as_secs_f64()<tl{
        let target_idx = *group_tree.t_nodes.iter().choose(&mut rng).unwrap();
        let mut child_idxs = group_tree.get_children_idxs(target_idx);
        eprintln!("1");
        let idx1 = *child_idxs.values().choose(&mut rng).unwrap().iter().choose(&mut rng).unwrap();
        if group_tree.groups[group_tree.groups[idx1].parent_group_idx.unwrap()].group_type=="t" || idx1==target_idx || group_tree.groups[idx1].member_idxs[0].len()!=1{
            continue;
        }
        eprintln!("2");
        if let Some(v) = child_idxs.get(&(group_tree.groups[idx1].end_day, group_tree.groups[idx1].member_idxs[0].len())){
            for &idx2 in v.iter(){
                if group_tree.groups[group_tree.groups[idx2].parent_group_idx.unwrap()].group_type=="t"{
                    continue;
                }
                eprintln!("joint {:?}", vec![idx1, idx2]);
                let undo_params = group_tree.forced_joint_t(input, &vec![idx1, idx2]);
                if group_tree.groups[group_tree.get_root_idx()].line_needed[W]>W{
                    eprintln!("undo joint");
                    group_tree.undo_forced_joint_t(input, &undo_params);
                } else {
                    eprintln!("join2: {:?}", vec![idx1, idx2]);
                    break;
                }
            }
        }
    }

    // group_tree.print_tree();
    let ans = group_tree.groups[group_tree.get_root_idx()].solve(input, (0, 0, W, W), &group_tree);
    ans
    // let mut init_state = State::init_state(input);
    // let mut best_state = simanneal(input, init_state, timer, 0.0);
    // best_state.print(input);
}
