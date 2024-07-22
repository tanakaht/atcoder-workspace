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
        // TODO: q/3のところを調整
        while n_small_groups*4<n && (n_small_groups as f64)*(n_small_groups as f64).log2()<((q/3) as f64){
            n_small_groups += d;
        }
        n_small_groups -= d;
        let mut small_groups = vec![HashSet::new(); n_small_groups];
        for i in 0..n{
            small_groups[i%n_small_groups].insert(i);
        }
        Self { measured_record: vec![], groups, small_groups, n, d, q, turn: 0, phase: 0 }
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

    fn get_sample_weight(&mut self, timer:&Instant, tl: f64) -> Vec<usize>{
        let mut rng = rand::thread_rng();
        let ts = timer.elapsed().as_secs_f64();
        let mut sample_weight = self.sample_exponentials(&mut rng, self.n);
        loop{
            if timer.elapsed().as_secs_f64()>tl{
                break;
            }
            let mut ng_cnt = 0;
            let mut measured_record = self.measured_record.clone();
            measured_record.shuffle(&mut rng);
            let mut change_weights = vec![0.0; self.n];
            let mut change_sum = 0;
            for (ls, rs, flg) in measured_record{
                let mut l = ls.iter().map(|&i| sample_weight[i]).sum::<usize>();
                let mut r = rs.iter().map(|&i| sample_weight[i]).sum::<usize>();
                if (flg==0 && l>r) || (flg==1 && l==r){
                    continue;
                }
                ng_cnt += 1;
                change_sum += (r-l)/(ls.len()+rs.len());
                // eprintln!("change_sum: {}", change_sum);
                for li in ls.iter(){
                    change_weights[*li] += 1.0;
                }
                for ri in rs.iter(){
                    change_weights[*ri] -= 1.0;
                }
            }
            if ng_cnt == 0{
                break;
            }
            eprintln!("ng_cnt: {}, cw: {:?}", ng_cnt, change_weights);
            let normal = Normal::new((change_sum as f64).sqrt(), (change_sum as f64).sqrt().sqrt()*0.0).unwrap();
            for (i, w) in change_weights.iter().enumerate(){
                let x = (sample_weight[i] as f64)+normal.sample(&mut rng)*w;
                // let x = (sample_weight[i] as f64)+w;
                // eprintln!("{}to{}", sample_weight[i], x);
                if x>=0.0{
                    sample_weight[i] = x as usize;
                } else {
                    sample_weight[i] = 0;
                }
            }
            eprint!("{} ", ng_cnt);
        }
        sample_weight
    }


    fn estimate(&mut self, timer:&Instant, tl: f64) -> (Vec<usize>, Vec<usize>){
        let mut sample_weights = vec![];
        loop{
            if timer.elapsed().as_secs_f64()>tl{
                break;
            }
            sample_weights.push(self.get_sample_weight(timer, tl));
        }
        let mut means = vec![];
        let mut vars = vec![];
        for i in 0..self.n{
            let mut sum = 0;
            for sample_idx in 0..sample_weights.len(){
                sum += sample_weights[sample_idx][i];
            }
            means.push(sum/sample_weights.len());
            let mut sum = 0;
            for sample_idx in 0..sample_weights.len(){
                sum += (sample_weights[sample_idx][i]-means[i])*(sample_weights[sample_idx][i]-means[i]);
            }
            vars.push(sum/sample_weights.len());
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

    fn make_group_based_on_small_group(&mut self, timer:&Instant, tl: f64){
        let sorted_idx = self.toporogical_sort_for_small_group();
        self.groups = vec![HashSet::new(); self.d];
        eprintln!("si: {:?}", sorted_idx);
        eprintln!("mr: {:?}", self.measured_record.iter().map(|(a, b, c)| c).collect_vec());
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
                    eprintln!("{:?}", sorted_idx);
                    for &item_idx in self.small_groups[sorted_idx[i*self.d+self.d-1-j]].iter(){
                        self.groups[j].insert(item_idx);
                    }
                }
            }
        }
    }

    fn solve(&mut self, timer:&Instant, tl: f64){
        let sorted_idxs = self.merge_sort_small_groups(&mut (0..self.small_groups.len()).collect_vec());
        for i in 0..(self.small_groups.len()-1){
            self.measured_record.push((self.small_groups[sorted_idxs[i+1]].iter().map(|&x| x).collect_vec(), self.small_groups[sorted_idxs[i]].iter().map(|&x| x).collect_vec(), 0));
        }

        while self.turn<self.q{
            self.move_small_group();
            self.make_group_based_on_small_group(timer, tl);
            self.print_group(true);
        }
        // let sample_weight = self.get_sample_weight(timer, tl);
        // eprintln!("{:?}", sample_weight);
        self.estimate(timer, tl);
        self.make_group_based_on_small_group(timer, tl);
        self.print_group(false);
    }

    fn compare_small_groups(&mut self, li: usize, ri: usize) -> char{
        self.measure(&self.small_groups[li].iter().map(|&x| x).collect_vec(), &self.small_groups[ri].iter().map(|&x| x).collect_vec())
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

    fn move_small_group(&mut self){
        let mut rng = rand::thread_rng();
        let sorted_idxs = self.toporogical_sort_for_small_group();
        let mut ls_idx = *sorted_idxs.first().unwrap();
        let mut ri = 1;
        let mut rs_idx = *sorted_idxs.last().unwrap();
        while self.small_groups[rs_idx].len()<=1{
            ri += 1;
            rs_idx = *sorted_idxs.get(sorted_idxs.len()-ri).unwrap();
        }
        let mut item_idx = *self.small_groups[rs_idx].iter().choose(&mut rng).unwrap();
        self.small_groups[ls_idx].insert(item_idx);
        self.small_groups[rs_idx].remove(&item_idx);
        let lrs = self.insert_small_group(vec![ls_idx, rs_idx]);
        for ((li, ri), idx) in lrs.iter().zip(vec![ls_idx, rs_idx].iter()){
            if let Some(li) = li{
                self.measured_record.push((self.small_groups[*idx].iter().map(|&x| x).collect_vec(), self.small_groups[*li].iter().map(|&x| x).collect_vec(), 0));
            }
            if let Some(ri) = ri{
                self.measured_record.push((self.small_groups[*ri].iter().map(|&x| x).collect_vec(), self.small_groups[*idx].iter().map(|&x| x).collect_vec(), 0));
            }
        }
    }

    fn insert_small_group(&mut self, idxs: Vec<usize>)->Vec<(Option<usize>, Option<usize>)>{
        let mut ret = vec![];
        let mut sorted_idxs = self.toporogical_sort_for_small_group();
        sorted_idxs.retain(|x| !idxs.contains(x));
        for idx in idxs.iter(){
            let mut li = 0;
            let mut ri = sorted_idxs.len();
            let mut mid;
            while ri-li>0{
                mid = (li+ri)/2;
                let flg = self.compare_small_groups(*idx, sorted_idxs[mid]);
                if flg=='<'{
                    ri = mid;
                } else {
                    li = mid+1;
                }
            }
            ret.push((sorted_idxs.get(li-1).map(|&x| x), sorted_idxs.get(li).map(|&x| x)));
            sorted_idxs.insert(li, *idx);
        }
        ret
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

    fn make_querys(&mut self) -> Vec<(Vec<usize>, Vec<usize>)>{
        // let mut rng = rand::thread_rng();
        // loop{
        //     let mut ls = vec![];
        //     let mut rs = vec![];
        //     let mut n = 1;
        //     if self.turn>self.q/2{
        //         n = 5;
        //     }
        //     for _ in 0..n{
        //         ls.push(rng.gen::<usize>()%self.n);
        //     }
        //     for _ in 0..n{
        //         rs.push(rng.gen::<usize>()%self.n);
        //     }
        //     if ls.iter().collect::<HashSet<_>>().intersection(&rs.iter().collect()).next().is_none() && ls.iter().collect::<HashSet<_>>().len()==ls.len() && rs.iter().collect::<HashSet<_>>().len()==rs.len(){
        //         return vec![(ls, rs)];
        //     }
        // }


        let mut rng = rand::thread_rng();
        if self.turn<3{
            // 0..3で最も軽いのを見つける
            let mut lrss = vec![(vec![0], vec![1]), (vec![0], vec![2]), (vec![1], vec![2])];
            return lrss;
        } else if self.turn<self.n{
            // 軽いのに目星つける
            let mut lrss = vec![];
            let minidx = most_frequent(self.measured_record.iter().take(3).map(|(ls, rs, flg)| rs[0]).collect_vec()).unwrap();
            for i in 3..self.n{
                lrss.push((vec![minidx], vec![i]));
            }
            return lrss;
        } else if self.turn<self.n+(self.q-self.n)/4{
            // 重いのでまずグループ分け
            // 適当に初期化
            if self.phase==0{
                let minidx = most_frequent(self.measured_record.iter().take(3).map(|(ls, rs, flg)| rs[0]).collect_vec()).unwrap();
                let mut heavys = HashSet::new();
                for (ls, rs, flg) in self.measured_record.iter(){
                    if rs[0]==minidx{
                        heavys.insert(ls[0]);
                    }
                }
                self.groups = vec![HashSet::new(); self.d];
                for (i, x) in heavys.iter().enumerate(){
                    self.groups[i%self.d].insert(*x);
                }
                self.phase +=1;
            }
            // toposo
            let sorted_idx = self.toporogical_sort_for_group();
            // 右から左へ渡す
            let ls_idx = *sorted_idx.first().unwrap();
            let rs_idx = *sorted_idx.last().unwrap();
            let item_idx = *self.groups[rs_idx].iter().choose(&mut rng).unwrap();
            self.groups[ls_idx].insert(item_idx);
            self.groups[rs_idx].remove(&item_idx);
            // 真ん中と比較させる
            let mut lrss = vec![];
            if self.d==2{
                lrss.push((self.groups[ls_idx].iter().map(|&x| x).collect_vec(), self.groups[rs_idx].iter().map(|&x| x).collect_vec()));
            } else {
                let mid_idx = *sorted_idx.get(self.d/2).unwrap();
                lrss.push((self.groups[mid_idx].iter().map(|&x| x).collect_vec(), self.groups[rs_idx].iter().map(|&x| x).collect_vec()));
                lrss.push((self.groups[mid_idx].iter().map(|&x| x).collect_vec(), self.groups[ls_idx].iter().map(|&x| x).collect_vec()));
            }
            return lrss;
        } else {
            // 軽いの含めてグループ分け
            // 軽いの求める
            let minidx = most_frequent(self.measured_record.iter().take(3).map(|(ls, rs, flg)| rs[0]).collect_vec()).unwrap();
            let mut lights = HashSet::new();
            lights.insert(minidx);
            for (ls, rs, flg) in self.measured_record.iter().take(self.n){
                if ls[0]==minidx{
                    lights.insert(ls[0]);
                }
            }
            // 適当に初期化
            if self.phase==1{
                for (i, x) in lights.iter().enumerate(){
                    self.groups[i%self.d].insert(*x);
                }
                self.phase +=1;
            }
            // toposo
            let sorted_idx = self.toporogical_sort_for_group();
            // 右から左へ渡す
            eprintln!("{:?}", sorted_idx);
            let ls_idx = *sorted_idx.first().unwrap();
            let rs_idx = *sorted_idx.last().unwrap();
            if let Some(&item_idx) = lights.intersection(&self.groups[rs_idx]).choose(&mut rng){
                self.groups[ls_idx].insert(item_idx);
                self.groups[rs_idx].remove(&item_idx);
            } else {
                let item_idx = *self.groups[rs_idx].iter().choose(&mut rng).unwrap();
                self.groups[ls_idx].insert(item_idx);
                self.groups[rs_idx].remove(&item_idx);
            }
            // 真ん中と比較させる
            let mut lrss = vec![];
            if self.d==2{
                lrss.push((self.groups[ls_idx].iter().map(|&x| x).collect_vec(), self.groups[rs_idx].iter().map(|&x| x).collect_vec()));
            } else {
                let mid_idx = *sorted_idx.get(self.d/2).unwrap();
                lrss.push((self.groups[mid_idx].iter().map(|&x| x).collect_vec(), self.groups[rs_idx].iter().map(|&x| x).collect_vec()));
                lrss.push((self.groups[mid_idx].iter().map(|&x| x).collect_vec(), self.groups[ls_idx].iter().map(|&x| x).collect_vec()));
            }
            return lrss;
        }
        vec![]
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
        // if flg_s=='>'{
        //     self.measured_record.push((ls.clone(), rs.clone(), 0));
        // } else if flg_s=='<'{
        //     self.measured_record.push((rs.clone(), ls.clone(), 0));
        // } else{
        //     self.measured_record.push((ls.clone(), rs.clone(), 1));
        // }
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
    let tl = 1.5;
    measure.solve(&timer, tl);
    eprintln!("{}", timer.elapsed().as_secs_f64());
}
