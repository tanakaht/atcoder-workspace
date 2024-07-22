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
                let idxs = daily_roots[d].iter().choose_multiple(&mut rng, 2);
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
                self.groups[parent_idx] = Group::joint_t(input, &child_idx.iter().map(|i| &self.groups[*i]).collect());
                self.groups[parent_idx].child_group_idx = child_idx;
                self.refresh(input, parent_idx);
            } else {
                let child_idx = self.groups[parent_idx].child_group_idx.clone();
                self.groups[parent_idx] = Group::joint_a(input, &self.groups[child_idx[0]], &self.groups[child_idx[1]]);
                self.groups[parent_idx].child_group_idx = child_idx;
                self.refresh(input, parent_idx);
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
        self.roots.remove(&idx);
        self.startdaywise_roots[self.groups[idx].start_day].remove(&idx);
        for d in self.groups[idx].start_day..self.groups[idx].end_day{
            self.daily_roots[d].remove(&idx);
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

    fn joint_a(&mut self, input: &Input, idx1: usize, idx2: usize){
        assert!(self.groups[idx1].start_day==self.groups[idx2].start_day);
        assert!(self.groups[idx1].end_day==self.groups[idx2].end_day);
        let mut new_group = Group::joint_a(input, &self.groups[idx1], &self.groups[idx2]);
        let new_group_idx = self.groups.len();
        self.groups[idx1].parent_group_idx = Some(new_group_idx);
        self.groups[idx2].parent_group_idx = Some(new_group_idx);
        new_group.child_group_idx.push(idx1);
        new_group.child_group_idx.push(idx2);
        self.groups.push(new_group);
        self.add_node(new_group_idx);
    }

    fn joint_t(&mut self, input: &Input, idxs: &Vec<usize>){
        let mut new_group = Group::joint_t(input, &idxs.iter().map(|i| &self.groups[*i]).collect());
        let new_group_idx = self.groups.len();
        for idx in idxs{
            self.groups[*idx].parent_group_idx = Some(new_group_idx);
            new_group.child_group_idx.push(*idx);
        }
        self.groups.push(new_group);
        self.add_node(new_group_idx);
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
        Self{start_day, end_day, member_idxs, member_sums, space, line_needed, child_group_idx: vec![], parent_group_idx: None, score: i64::MAX, group_type: "leaf".to_string()}
    }

    fn update_line_needed(&mut self, input: &Input, group_tree: &GroupTree){
        if self.group_type=="leaf"{
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

    fn joint_t(input: &Input, gs: &Vec<&Group>) -> Group{
        for i in 0..gs.len()-1{
            assert!(gs[i].end_day==gs[i+1].start_day);
            assert!(gs[i].member_idxs[0].len()==gs[i+1].member_idxs[0].len());
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
        Self{start_day: gs[0].start_day, end_day: gs.last().unwrap().end_day, member_idxs: new_member_idxs, member_sums, space, line_needed, child_group_idx: vec![], parent_group_idx: None, score: i64::MAX, group_type: "t".to_string()}
    }

    fn joint_a(input: &Input, g1: &Group, g2: &Group) -> Group{
        assert!(g1.start_day==g2.start_day);
        assert!(g1.end_day==g2.end_day);
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
        Self{start_day: g1.start_day, end_day: g1.end_day, member_idxs: new_member_idxs, member_sums, space, line_needed, child_group_idx: vec![], parent_group_idx: None, score: i64::MAX, group_type: "a".to_string()}
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
                let h1 = group_tree.groups[group1_idx].line_needed[w]-1;
                ans1 = group_tree.groups[group1_idx].solve(input, (rect.0, rect.1, rect.2, rect.1+h1), group_tree);
                ans2 = group_tree.groups[group2_idx].solve(input, (rect.0, rect.1+h1, rect.2, rect.3), group_tree);
            } else {
                let w1 = group_tree.groups[group1_idx].line_needed[h]-1;
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
    for d in 0..input.D{
        while group_tree.daily_roots[d].len()>1{
            let idxs = group_tree.daily_roots[d].iter().choose_multiple(&mut rng, 2);
            group_tree.joint_a(input, *idxs[0], *idxs[1]);
        }
    }
    let mut idxs = vec![];
    for d in 0..input.D{
        idxs.push(*group_tree.startdaywise_roots[d].iter().next().unwrap());
    }
    group_tree.joint_t(input, &idxs);
    // for d in 1..input.D{
    //     let idx1 = *group_tree.startdaywise_roots[0].iter().next().unwrap();
    //     let idx2 = *group_tree.startdaywise_roots[d].iter().next().unwrap();
    //     // eprintln!("0: {} {}",  group_tree.groups[idx1].space, group_tree.groups[idx1].blank_space);
    //     // eprintln!("{}: {} {}",  d, group_tree.groups[idx2].space, group_tree.groups[idx2].blank_space);
    //     group_tree.joint_t(input, &vec![idx1, idx2]);
    // }
    // eprintln!("{:?} {}", group_tree.groups[*group_tree.roots.iter().next().unwrap()].space, group_tree.groups[*group_tree.roots.iter().next().unwrap()].blank_space);
    let ans = group_tree.groups[*group_tree.roots.iter().next().unwrap()].solve(input, (0, 0, W, W), &group_tree);
    ans
    // let mut init_state = State::init_state(input);
    // let mut best_state = simanneal(input, init_state, timer, 0.0);
    // best_state.print(input);
}
