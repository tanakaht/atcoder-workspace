#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap, BinaryHeap, VecDeque};
use std::process::exit;
use itertools::Itertools;
use libm::ceil;
use proconio::{input, source::line::LineSource};
use rand_distr::Normal;
use std::io::{stdin, stdout, BufReader, Write};
use std::fmt::Display;
use std::cmp::{Reverse, Ordering, max, min};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;
use ndarray::prelude::*;


fn is_subset(base: &HashSet<usize>, other: &HashSet<usize>) -> bool{
    base.intersection(&other).count()==other.len()
}

fn most_frequent(items: Vec<usize>) -> Option<usize> {
    let mut occurrences: HashMap<usize, usize> = HashMap::new();
    for item in items.iter() {
        *occurrences.entry(item.clone()).or_insert(0) += 1;
    }
    occurrences.into_iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(key, _)| key)
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Measure{
    measured_record: Vec<(Vec<usize>, Vec<usize>, usize)>, // (l, r, flg). 基本, l>r. flg=1ならl=r
    groups: Vec<HashSet<usize>>,
    small_groups: Vec<HashSet<usize>>,
    small_groups_history: Vec<Vec<HashSet<usize>>>,
    sample_weights: Vec<(usize, Vec<usize>)>,
    n: usize,
    d: usize,
    q: usize,
    turn: usize,
    phase: usize,
}

impl Measure{
    fn new() -> Self{
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            n: usize,
            d: usize,
            q: usize
        }
        let mut groups = vec![HashSet::new(); d];
        for i in 0..n{
            groups[i%d].insert(i);
        }
        let mut n_small_groups = 2*d;
        if  (n as f64)*(n as f64).log2()<((q/2) as f64){
            while n_small_groups*4<n && (n_small_groups as f64)*(n_small_groups as f64).log2()<((q/2) as f64)-(n as f64)*(n as f64).log2(){
                n_small_groups += d;
            }
            n_small_groups -= d;

        } else {
            // TODO: q/3のところを調整
            while n_small_groups*4<n && (n_small_groups as f64)*(n_small_groups as f64).log2()<((q/2) as f64){
                n_small_groups += d;
            }
            n_small_groups -= d;
        }
        let mut small_groups = vec![HashSet::new(); n_small_groups];
        for i in 0..n{
            small_groups[i%n_small_groups].insert(i);
        }
        let mut ret = Self { measured_record: vec![], groups, small_groups, small_groups_history: vec![], sample_weights: vec![], n, d, q, turn: 0, phase: 0 };
        let mut rng = rand::thread_rng();
        for i in 0..100{
            ret.sample_weights.push((1000000000000, ret.sample_exponentials(&mut rng, n)));
        }
        ret
    }

    fn sample_exponentials(&self, rng: &mut ThreadRng, n: usize) -> Vec<usize>{
        let lambda = 100000.0;
        let max_v = (lambda*(self.n as f64)/(self.d as f64)) as usize;
        let mut ret = vec![];
        for _ in 0..n{
            loop {
                let x = rng.gen::<f64>()%1.0;
                let y = (-lambda * (1.0-x).ln()) as usize;
                if y<max_v{
                    ret.push(y);
                    break;
                }
            }
        }
        ret
    }

    fn get_change_weight_for_sample_weight(&self, sample_weight: &Vec<usize>) -> (f64, Vec<f64>){
        let mut change_sum = 0.0;
        let mut change_weights = vec![0.0; self.n];
        change_sum = 0.0;
        for (ls, rs, flg) in self.measured_record.iter(){
            let mut l = ls.iter().map(|&i| sample_weight[i]).sum::<usize>();
            let mut r = rs.iter().map(|&i| sample_weight[i]).sum::<usize>();
            if (*flg==0 && l>r) || (*flg==1 && l==r){
                continue;
            }
            change_sum += ((r-l) as f64)/((ls.len()+rs.len()) as f64);
            // eprintln!("change_sum: {}", change_sum);
            for li in ls.iter(){
                change_weights[*li] += 1.0;
            }
            for ri in rs.iter(){
                change_weights[*ri] -= 1.0;
            }
        }
        (change_sum, change_weights)
    }

    fn mod_sample_weight(&self, base: &Vec<usize>, timer:&Instant, tl: f64) -> (usize, Vec<usize>){
        let mut rng = rand::thread_rng();
        let ts = timer.elapsed().as_secs_f64();
        let mut sample_weight = base.clone();
        let mut cnt = 0;
        let mut ng_cnt = 0;
        let mut change_sum: f64 = 1000000000000.0;
        let mut change_weights = vec![0.0; self.n];
        loop{
            cnt += 1;
            (change_sum, change_weights) = self.get_change_weight_for_sample_weight(&sample_weight);
            if change_sum<0.1{
                break;
            }
            if timer.elapsed().as_secs_f64()>tl{
                break;
            }
            let normal = Normal::new((change_sum as f64).sqrt(), (change_sum as f64).sqrt().sqrt()).unwrap();
            for (i, w) in change_weights.iter().enumerate(){
                let x = (sample_weight[i] as f64)+normal.sample(&mut rng)*w;
                // let x = (sample_weight[i] as f64)+w*(change_sum as f64).sqrt();
                // let x = (sample_weight[i] as f64)+w;
                // eprintln!("{}to{}", sample_weight[i], x);
                if x>=0.0{
                    sample_weight[i] = x as usize;
                } else {
                    sample_weight[i] = 0;
                }
            }
            // eprint!("{} ", ng_cnt);
        }
        // eprintln!("loop_cnt: {}, ng_cnt: {}, change_sum: {}", cnt, ng_cnt, change_sum);
        (change_sum as usize, sample_weight)
    }

    fn estimate(&mut self, timer:&Instant, tl: f64) -> (Vec<usize>, Vec<usize>){
        let mut rng = rand::thread_rng();
        for i in 0..self.sample_weights.len(){
            let (change_sum, sample_weight) = self.mod_sample_weight(&self.sample_weights[i].1, timer, tl);
            self.sample_weights[i] = (change_sum, sample_weight);
        }
        let mut sample_idxs = vec![];
        for i in 0..self.sample_weights.len(){
            if self.sample_weights[i].0<100 || sample_idxs.len()==0{
                sample_idxs.push(i);
            }
        }
        eprintln!("turn: {}, n_sample: {}", self.turn, sample_idxs.len());
        let mut means = vec![];
        let mut vars = vec![];
        for i in 0..self.n{
            let mut sum = 0;
            for sample_idx in sample_idxs.iter(){
                sum += self.sample_weights[*sample_idx].1[i];
            }
            means.push(sum/sample_idxs.len());
            let mut sum = 0;
            for sample_idx in sample_idxs.iter(){
                sum += (self.sample_weights[*sample_idx].1[i]-means[i])*(self.sample_weights[*sample_idx].1[i]-means[i]);
            }
            vars.push(sum/sample_idxs.len());
        }
        (means, vars)
    }

    fn make_group_based_on_estimate(&mut self, timer:&Instant, tl: f64){
        let (means, vars) = self.estimate(timer, tl);
        let mut sorted_idx = (0..self.n).collect::<Vec<usize>>();
        sorted_idx.sort_by(|&i, &j| means[i].cmp(&means[j]));
        let mut bh: BinaryHeap<(Reverse<usize>, Vec<usize>)> = BinaryHeap::new();
        for i in 0..self.d{
            bh.push((Reverse(0), vec![]));
        }
        for idx in sorted_idx.iter(){
            if let Some((Reverse(weight_sum), mut group)) = bh.pop(){
                group.push(*idx);
                bh.push((Reverse(weight_sum+means[*idx]), group));
            }
        }
        let mut ans = vec![0; self.n];
        for (i, (_, idxs)) in bh.iter().enumerate(){
            self.groups[i] = idxs.iter().map(|&x| x).collect();
        }
    }

    fn make_group_based_on_small_group_history(&mut self, timer:&Instant, tl: f64){
        let (means, vars) = self.estimate(timer, tl);
        let mut best_idx = 0;
        let mut best_score = 1000000000000.0;
        for (i, small_groups) in self.small_groups_history.iter().enumerate(){
            let mut group_w = vec![0; small_groups.len()];
            for (j, small_group) in small_groups.iter().enumerate(){
                group_w[j] = small_group.iter().map(|&i| means[i]).sum::<usize>();
            }
            let mut score = 0.0;
            let groups_means = group_w.iter().map(|&w| w).sum::<usize>()/group_w.len();
            for w in group_w.iter(){
                score += (*w as f64-groups_means as f64).abs();
            }
            if score<best_score{
                best_score = score;
                best_idx = i;
            }
        }
        self.small_groups = self.small_groups_history[best_idx].clone();
        self.make_group_based_on_small_group(timer, tl);
    }

    fn make_group_based_on_small_group(&mut self, timer:&Instant, tl: f64){
        let sorted_idx = self.toporogical_sort_for_small_group();
        self.groups = vec![HashSet::new(); self.d];
        for i in 0..(self.small_groups.len()/self.d){
            // 正順か逆順か
            if (i%2==0) && i!=self.small_groups.len()/self.d-1{
                for j in 0..self.d{
                    for &item_idx in self.small_groups[sorted_idx[i*self.d+j]].iter(){
                        self.groups[j].insert(item_idx);
                    }
                }
            } else {
                for j in 0..self.d{
                    for &item_idx in self.small_groups[sorted_idx[i*self.d+self.d-1-j]].iter(){
                        self.groups[j].insert(item_idx);
                    }
                }
            }
        }
    }

    fn solve(&mut self, timer:&Instant, tl: f64){
        if  (self.n as f64)*(self.n as f64).log2()<((self.q/2) as f64){
            let sorted_idxs = self.merge_sort(&mut (0..self.n).collect_vec());
            for i in 0..(self.small_groups.len()-1){
                self.measured_record.push((vec![sorted_idxs[i+1]], vec![sorted_idxs[i+1]], 0));
            }
        }
        let sorted_idxs = self.merge_sort_small_groups(&mut (0..self.small_groups.len()).collect_vec());
        for i in 0..(self.small_groups.len()-1){
            self.measured_record.push((self.small_groups[sorted_idxs[i+1]].iter().map(|&x| x).collect_vec(), self.small_groups[sorted_idxs[i]].iter().map(|&x| x).collect_vec(), 0));
        }

        while self.turn<self.q{
            self.move_small_group(timer, tl*(self.turn as f64)/(self.q as f64));
            self.make_group_based_on_small_group(timer, tl);
            self.print_group(true);
            // self.estimate(timer, tl*(self.turn as f64)/(self.q as f64));
        }
        // let sample_weight = self.get_sample_weight(timer, tl);
        // eprintln!("{:?}", sample_weight);
        self.estimate(timer, tl);
        self.make_group_based_on_small_group_history(timer, tl);
        // self.make_group_based_on_small_group(timer, tl);
        // self.make_group_based_on_small_group(timer, tl);
        self.print_group(false);
    }

    fn compare_small_groups(&mut self, li: usize, ri: usize) -> char{
        self.measure(&self.small_groups[li].iter().map(|&x| x).collect_vec(), &self.small_groups[ri].iter().map(|&x| x).collect_vec())
    }

    fn merge_sort(&mut self, idxs: &Vec<usize>) -> Vec<usize>{
        if idxs.len()==1{
            return idxs.clone();
        }
        let mid = idxs.len()/2;
        let left = self.merge_sort(&idxs[0..mid].to_vec());
        let right = self.merge_sort(&idxs[mid..].to_vec());
        let mut ret = vec![];
        let mut i = 0;
        let mut j = 0;
        while i<left.len() && j<right.len(){
            if self.measure(&vec![left[i]], &vec![right[j]])=='<'{
                ret.push(left[i]);
                i += 1;
            } else {
                ret.push(right[j]);
                j += 1;
            }
        }
        while i<left.len(){
            ret.push(left[i]);
            i += 1;
        }
        while j<right.len(){
            ret.push(right[j]);
            j += 1;
        }
        ret
    }

    fn merge_sort_small_groups(&mut self, idxs: &Vec<usize>) -> Vec<usize>{
        if idxs.len()==1{
            return idxs.clone();
        }
        let mid = idxs.len()/2;
        let left = self.merge_sort_small_groups(&idxs[0..mid].to_vec());
        let right = self.merge_sort_small_groups(&idxs[mid..].to_vec());
        let mut ret = vec![];
        let mut i = 0;
        let mut j = 0;
        while i<left.len() && j<right.len(){
            if self.compare_small_groups(left[i], right[j])=='<'{
                ret.push(left[i]);
                i += 1;
            } else {
                ret.push(right[j]);
                j += 1;
            }
        }
        while i<left.len(){
            ret.push(left[i]);
            i += 1;
        }
        while j<right.len(){
            ret.push(right[j]);
            j += 1;
        }
        ret
    }

    fn move_small_group(&mut self, timer:&Instant, tl: f64){
        let mut rng = rand::thread_rng();
        let sorted_idxs = self.toporogical_sort_for_small_group();
        let mut ls_idx = *sorted_idxs.first().unwrap();
        let mut ri = 1;
        let mut rs_idx = *sorted_idxs.last().unwrap();
        while self.small_groups[rs_idx].len()<=1{
            ri += 1;
            rs_idx = *sorted_idxs.get(sorted_idxs.len()-ri).unwrap();
        }
        let (means, vars) = self.estimate(timer, tl);
        let mut best_set = HashSet::new();
        let mut best_score = 1000000000000.0;
        let mut idxs = self.small_groups[ls_idx].iter().map(|&i| i).collect_vec();
        let mut ls_sum = self.small_groups[ls_idx].iter().map(|&i| means[i]).sum::<usize>();
        let mut rs_sum = self.small_groups[rs_idx].iter().map(|&i| means[i]).sum::<usize>();
        let mut appeared = HashSet::new();
        loop{
            best_set = HashSet::new();
            best_score = 1000000000000.0;
            ls_sum = self.small_groups[ls_idx].iter().map(|&i| means[i]).sum::<usize>();
            rs_sum = self.small_groups[rs_idx].iter().map(|&i| means[i]).sum::<usize>();
            idxs = self.small_groups[ls_idx].iter().map(|&i| i).collect_vec();
            idxs.append(&mut self.small_groups[rs_idx].iter().map(|&i| i).collect_vec());
            for flg in 0..(1<<idxs.len()){
                let mut s = HashSet::new();
                for (i, idx) in idxs.iter().enumerate(){
                    if flg&(1<<i)>0{
                        s.insert(*idx);
                    }
                }
                let score = (((s.iter().map(|&i| means[i]).sum::<usize>() as f64))-(((ls_sum+rs_sum)/2) as f64)).abs();
                if score<best_score{
                    best_score = score;
                    best_set = s;
                }
            }
            if best_set.iter().map(|&i| means[i]).sum::<usize>()!=ls_sum && best_set.iter().map(|&i| means[i]).sum::<usize>()!=rs_sum{
                break;
            }
            appeared.insert((ls_idx, rs_idx));
            if appeared.len()==sorted_idxs.len()*(sorted_idxs.len()-1)/2{
                break;
            }
            ls_idx = *sorted_idxs.iter().choose(&mut rng).unwrap();
            rs_idx = *sorted_idxs.iter().choose(&mut rng).unwrap();
            while ls_idx==rs_idx || appeared.contains(&(ls_idx, rs_idx)){
                ls_idx = *sorted_idxs.iter().choose(&mut rng).unwrap();
                rs_idx = *sorted_idxs.iter().choose(&mut rng).unwrap();
            }
        }
        eprintln!("m: {}, ls: {}, rs: {}", best_set.iter().map(|&i| means[i]).sum::<usize>(), ls_sum, rs_sum);
        // let mut item_idx = *self.small_groups[rs_idx].iter().choose(&mut rng).unwrap();
        self.small_groups[ls_idx] = idxs.iter().filter(|&x| !best_set.contains(x)).map(|&x| x).collect();
        self.small_groups[rs_idx] = best_set;
        self.insert_small_group(vec![ls_idx, rs_idx]);
        if self.turn<self.q{
            self.small_groups_history.push(self.small_groups.clone());
        }
    }

    fn insert_small_group(&mut self, idxs: Vec<usize>){
        let mut sorted_idxs = self.toporogical_sort_for_small_group();
        sorted_idxs.retain(|x| !idxs.contains(x));
        for idx in idxs.iter(){
            if self.turn==self.q{
                return;
            }
            let mut li = 0;
            let mut ri = sorted_idxs.len();
            let mut mid;
            while ri-li>0{
                if self.turn==self.q{
                    return;
                }
                    mid = (li+ri)/2;
                let flg = self.compare_small_groups(*idx, sorted_idxs[mid]);
                if flg=='<'{
                    ri = mid;
                } else {
                    li = mid+1;
                }
            }
            if li>0{
                self.measured_record.push((self.small_groups[*idx].iter().map(|&x| x).collect_vec(), self.small_groups[sorted_idxs[li-1]].iter().map(|&x| x).collect_vec(), 0));
            }
            if li<sorted_idxs.len(){
                self.measured_record.push((self.small_groups[sorted_idxs[li]].iter().map(|&x| x).collect_vec(), self.small_groups[*idx].iter().map(|&x| x).collect_vec(), 0));
            }
            eprintln!("insert_to: {}, len: {}", li, sorted_idxs.len());
            sorted_idxs.insert(li, *idx);
        }
    }

    fn toporogical_sort_for_small_group(&self) -> Vec<usize>{
        let mut g = vec![HashSet::new(); self.small_groups.len()];
        let mut g_rev = vec![HashSet::new(); self.small_groups.len()];
        let mut same = vec![vec![]; self.small_groups.len()];
        for (ls, rs, flg)  in self.measured_record.iter(){
            let ls_set = ls.iter().map(|&x| x).collect::<HashSet<_>>();
            let rs_set = rs.iter().map(|&x| x).collect::<HashSet<_>>();
            for ls_idx in 0..self.small_groups.len(){
                if !is_subset(&self.small_groups[ls_idx], &ls_set){
                    continue;
                }
                for rs_idx in 0..self.small_groups.len(){
                    if ls_idx==rs_idx || !is_subset(&rs_set, &self.small_groups[rs_idx]){
                        continue;
                    }
                    if *flg==1{
                        // TODO: 対応
                        // same[ls_idx].push(rs_idx);
                        // same[rs_idx].push(ls_idx);
                        continue;
                    }
                    g[ls_idx].insert(rs_idx);
                    g_rev[rs_idx].insert(ls_idx);
                }
            }
        }
        let mut ret = vec![];
        let mut q = VecDeque::new();
        for i in 0..self.small_groups.len(){
            if g[i].is_empty(){
                q.push_back(i);
            }
        }
        while !q.is_empty(){
            let i = q.pop_front().unwrap();
            ret.push(i);
            for &j in g_rev[i].iter(){
                g[j].remove(&i);
                if g[j].is_empty(){
                    q.push_back(j);
                }
            }
            for &j in same[i].iter(){
                if !ret.contains(&j){
                    q.push_front(j);
                }
            }
        }
        ret
    }

    fn toporogical_sort_for_group(&self) -> Vec<usize>{
        let mut g = vec![HashSet::new(); self.d];
        let mut g_rev = vec![HashSet::new(); self.d];
        let mut same = vec![vec![]; self.d];
        for (ls, rs, flg)  in self.measured_record.iter(){
            let ls_set = ls.iter().map(|&x| x).collect::<HashSet<_>>();
            let rs_set = rs.iter().map(|&x| x).collect::<HashSet<_>>();
            for ls_idx in 0..self.d{
                if !is_subset(&self.groups[ls_idx], &ls_set){
                    continue;
                }
                for rs_idx in 0..self.d{
                    if ls_idx==rs_idx || !is_subset(&rs_set, &self.groups[rs_idx]){
                        continue;
                    }
                    if *flg==1{
                        same[ls_idx].push(rs_idx);
                        same[rs_idx].push(ls_idx);
                        continue;
                    }
                    g[ls_idx].insert(rs_idx);
                    g_rev[rs_idx].insert(ls_idx);
                }
            }
        }
        let mut ret = vec![];
        let mut q = VecDeque::new();
        for i in 0..self.d{
            if g[i].is_empty(){
                q.push_back(i);
            }
        }
        while !q.is_empty(){
            let i = q.pop_front().unwrap();
            ret.push(i);
            for &j in g_rev[i].iter(){
                g[j].remove(&i);
                if g[j].is_empty(){
                    q.push_back(j);
                }
            }
            for &j in same[i].iter(){
                if !ret.contains(&j){
                    q.push_front(j);
                }
            }
        }
        ret
    }

    fn print_group(&self, debug: bool){
        let mut ans = vec![0; self.n];
        for (i, group) in self.groups.iter().enumerate(){
            for &idx in group.iter(){
                ans[idx] = i;
            }
        }
        if debug{
            println!("#c {}", ans.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        } else {
            println!("{}", ans.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        }
    }

    fn measure(&mut self, ls: &Vec<usize>, rs: &Vec<usize>) -> char{
        if self.turn == self.q{
            return '.';
        }
        // TODO: lsとrsのintersectionが空であることチェック
        assert!(ls.iter().collect::<HashSet<_>>().intersection(&rs.iter().collect()).next().is_none(), "ls: {:?}, rs: {:?}", ls, rs);
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        let s = format!("{} {}", ls.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "), rs.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        println!("{} {} {}", ls.len(), rs.len(), s);
        input!{
            from &mut source,
            flg_s: char
        }
        self.turn += 1;
        // self.estimate();
        eprintln!("turn: {} end", self.turn);
        // self.print_group(true);
        flg_s
    }

    fn measure_and_record(&mut self, ls: &Vec<usize>, rs: &Vec<usize>) -> char{
        if self.turn == self.q{
            return '.';
        }
        // TODO: lsとrsのintersectionが空であることチェック
        assert!(ls.iter().collect::<HashSet<_>>().intersection(&rs.iter().collect()).next().is_none(), "ls: {:?}, rs: {:?}", ls, rs);
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        let s = format!("{} {}", ls.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "), rs.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        println!("{} {} {}", ls.len(), rs.len(), s);
        input!{
            from &mut source,
            flg_s: char
        }
        if flg_s=='>'{
            self.measured_record.push((ls.clone(), rs.clone(), 0));
        } else if flg_s=='<'{
            self.measured_record.push((rs.clone(), ls.clone(), 0));
        } else{
            self.measured_record.push((ls.clone(), rs.clone(), 1));
        }
        // let mut new_sample_weights = vec![];
        // for sample_weight in self.sample_weights.iter(){
        //     let mut l = ls.iter().map(|&i| sample_weight[i]).sum::<usize>();
        //     let mut r = rs.iter().map(|&i| sample_weight[i]).sum::<usize>();
        //     if (flg_s=='>' && l>r) || (flg_s=='<' && l<r) || (flg_s=='=' && l==r){
        //         new_sample_weights.push(sample_weight.clone());
        //     }
        // }
        // self.sample_weights = new_sample_weights;
        self.turn += 1;
        // self.estimate();
        eprintln!("turn: {} end", self.turn);
        // self.print_group(true);
        flg_s
    }

}


fn main() {
    let timer = Instant::now();
    let mut measure = Measure::new();
    let tl = 1.2;
    measure.solve(&timer, tl);
    eprintln!("{}", timer.elapsed().as_secs_f64());
    // assert!(timer.elapsed().as_secs_f64()<1.85, "time over");
}
