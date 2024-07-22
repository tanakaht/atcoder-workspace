#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::HashSet;
use std::process::exit;
use itertools::Itertools;
use num::{range, ToPrimitive};
use proconio::*;
use std::iter::FromIterator;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fmt::Display;
use std::cmp::{Reverse, Ordering};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;

const INF: usize = 1 << 31;
const DEFAULTDIST: usize = 1000000000;
const SEED: u128 = 42;
const N:usize = 200;

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



struct BIT {
    data: Vec<usize>
}

impl BIT{
    fn new(n: usize) -> BIT {
        BIT { data: vec![0; n + 1] }
    }

    fn sum(&self, i: usize) -> usize {
        let mut r = 0;
        let mut i = i;
        while i > 0 {
            r += self.data[i];
            i &= i - 1;
        }
        r
    }
    fn add(&mut self, i: usize, value: usize) {
        let mut i = i + 1;
        while i < self.data.len() {
            self.data[i] += value.clone();
            i += i & (!i + 1);
        }
    }
}

fn inversion(xs: &Vec<usize>) -> usize {
    if xs.is_empty(){
        return 0;
    }
    let max_ele = *xs.iter().max().unwrap();
    let mut bit = BIT::new(max_ele+1);
    xs.iter().rev().fold(0, |r, x|{
        bit.add(*x, 1);
        r + bit.sum(*x)
    })
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Load {
    w: i32,
    h: i32,
    d: i32,
    idx: usize,
    f: bool,
    g: bool,
}

impl Load{
    fn rotated(&self, flg: usize)->Load{
        match flg {
            0 => Load{w: self.w, h: self.h, d: self.d, idx: self.idx, f: self.f, g: self.g},
            1 => Load{w: self.h, h: self.w, d: self.d, idx: self.idx, f: self.f, g: self.g},
            2 => Load{w: self.d, h: self.h, d: self.w, idx: self.idx, f: self.f, g: self.g},
            3 => Load{w: self.h, h: self.d, d: self.w, idx: self.idx, f: self.f, g: self.g},
            4 => Load{w: self.d, h: self.w, d: self.h, idx: self.idx, f: self.f, g: self.g},
            _ => Load{w: self.w, h: self.d, d: self.h, idx: self.idx, f: self.f, g: self.g}
        }
    }

    fn bottom_area(&self) -> i32{
        self.w*self.h
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct PlacedLoad {
    load: Load,
    xs: i32,
    ys: i32,
    zs: i32,
    xe: i32,
    ye: i32,
    ze: i32,
    r: usize,
    is_dummy: bool
}

impl PlacedLoad{
    /// dirについて、xyzrが以下のどれに合うか
    /// 0: (xs, ys), 1(xs, ye), 2(xe, ys), 3 (xe, ye)
    fn new(base_load: &Load, xyzr: (i32, i32, i32, usize), is_dummy: bool, dir: usize) -> Self{
        let (xs, ys, zs, r) = xyzr;
        let load = base_load.rotated(r);
        let (w, h, d) = (load.w, load.h, load.d);
        match dir{
            0 => {Self{load, xs, ys, zs, xe: xs+w, ye: ys+h, ze: zs+d, r, is_dummy}},
            1 => {Self{load, xs, ys: ys-h, zs, xe: xs+w, ye: ys, ze: zs+d, r, is_dummy}},
            2 => {Self{load, xs: xs-w, ys, zs, xe: xs, ye: ys+h, ze: zs+d, r, is_dummy}},
            _ => {Self{load, xs: xs-w, ys: ys-h, zs, xe: xs, ye: ys, ze: zs+d, r, is_dummy}},
        }
    }

    fn bottom_area(&self) -> i32{
        self.load.bottom_area()
    }

    /// 0: xで接触(self<oppo)
    /// 1: xで接触(self>oppo)
    /// 2: yで接触(self<oppo)
    /// 3: yで接触(self>oppo)
    /// 4: zで接触(self<oppo)
    /// 5: zで接触(self>oppo)
    /// 6: 接点なし だが上下にある (self<oppo)
    /// 7: 接点なし だが上下にある (self>oppo)
    /// 8: 共有領域あり
    /// 9: 接点なし(その他)
    fn common_info(&self, oppo: &PlacedLoad) -> usize{
        let x_common = min!(self.xe, oppo.xe) > max!(self.xs, oppo.xs);
        let y_common = min!(self.ye, oppo.ye) > max!(self.ys, oppo.ys);
        let z_common = min!(self.ze, oppo.ze) > max!(self.zs, oppo.zs);
        if (x_common && y_common && z_common){
            return 8;
        } else if (min!(self.xe, oppo.xe)==max!(self.xs, oppo.xs) && y_common && z_common){
            return if(self.xs<oppo.xs){0} else {1};
        } else if (min!(self.ye, oppo.ye)==max!(self.ys, oppo.ys) && x_common && z_common){
            return if(self.ys<oppo.ys){2} else {3};
        } else if (min!(self.ze, oppo.ze)==max!(self.zs, oppo.zs) && x_common && y_common){
            return if(self.zs<oppo.zs){4} else {5};
        } else if (x_common && y_common){
            return if(self.zs<oppo.zs){6} else {7};
        } else {
            return 9
        }
    }

    fn common_xy_area(&self, oppo: &PlacedLoad) -> i32{
        let common_info =  self.common_info(oppo);
        match common_info{
            4 | 5 | 6 | 7 | 8 => {
                let x_diff = min!(self.xe, oppo.xe) - max!(self.xs, oppo.xs);
                let y_diff = min!(self.ye, oppo.ye) - max!(self.ys, oppo.ys);
                return x_diff*y_diff
            }
            _ => {return 0}
        }
    }

    fn can_place(&self, oppos: &Vec<PlacedLoad>) -> bool{
        // 他のloadとぶつからない、かつ底面積が十分ある
        let mut bottom_area = if self.zs==0 {self.bottom_area()} else {0};
        for oppo in oppos.iter(){
            let common_info = self.common_info(oppo);
            match common_info{
                4 | 6 | 8 => {return false},
                5 => {bottom_area += self.common_xy_area(oppo)},
                _ => {}
            }
        }
        bottom_area*10>=self.bottom_area()*6
    }
}




#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    M: usize,
    W: i32,
    H: i32,
    B: i32,
    D: i32,
    loads: Vec<Load>,
    load_type: Vec<Load>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            M: usize,
            W: i32,
            H: i32,
            B: i32,
            D: i32,
        }
        let mut loads: Vec<Load> = vec![];
        let mut load_type: Vec<Load> = vec![];
        for idx in 0..M{
            input!{
                h: i32,
                w: i32,
                d: i32,
                a: usize,
                f_: char,
                g_: char,
            };
            for _ in 0..a{
                let load = Load{h, w, d, idx, f: f_=='Y', g: g_=='Y'};
                loads.push(load)
            }
            let load = Load{h, w, d, idx, f: f_=='Y', g: g_=='Y'};
            load_type.push(load)
        }
        Self { M, W, H, B, D, loads, load_type}
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Field{
    placedloads: Vec<PlacedLoad>,
    bl_stables: Vec<((i32, i32), usize)>,
    N: usize,
    score: Option<usize>,
    D: i32,
    rest_load: Vec<usize>
}

impl Field{
    fn new(input: &Input) -> Self{
        let base = Load{ w: input.W, h: input.H, d: 1, idx: 1000, f: false, g: true };
        let tate = Load{ w: input.W, h: 1, d: input.D*10, idx: 1001, f: false, g: false };
        let yoko = Load{ w: 1, h: input.H, d: input.D*10, idx: 1002, f: false, g: false };
        let block = Load{ w: input.B, h: input.B, d: input.D*10, idx: 1003, f: false, g: false };
        let placedloads = vec![PlacedLoad::new(&base, (0, 0, -1, 0), true, 0),
                                        PlacedLoad::new(&tate, (0, -1, 0, 0), true, 0),
                                        PlacedLoad::new(&tate, (0, input.H, 0, 0), true, 0),
                                        PlacedLoad::new(&yoko, (-1, 0, 0, 0), true, 0),
                                        PlacedLoad::new(&yoko, (input.W, 0, 0, 0), true, 0),
                                        PlacedLoad::new(&block, (0, 0, 0, 0), true, 0),
                                        PlacedLoad::new(&block, (0, input.H, 0, 0), true, 1),
                                        PlacedLoad::new(&block, (input.W, 0, 0, 0), true, 2),
                                        PlacedLoad::new(&block, (input.W, input.H, 0, 0), true, 3),
                                        ];
        let mut rest_load = vec![0; input.M];
        for load in input.loads.iter(){
            rest_load[load.idx] += 1;
        }
        let mut ret = Self {placedloads: vec![], bl_stables: vec![], N: input.loads.len(), score: None, D: input.D, rest_load};
        for pl in placedloads.into_iter(){
            ret.add_placedload(pl)
        }
        ret
    }

    fn get_all_neighbor(&self, input: &Input) -> Vec<Field>{
        let mut ret = vec![];
        for (idx, cnt) in self.rest_load.iter().enumerate(){
            if *cnt!=0{
                let load = &input.load_type[idx];
                let mut neighbnors =  self.get_neighbor(input, load, 100);
                ret.append(&mut neighbnors);
            }
        }
        ret
    }

    fn get_neighbor(&self, input: &Input, load: &Load, n: usize) -> Vec<Field>{
        let mut ret = vec![];
        let mut rng = rand::thread_rng();
        let mut kouho: Vec<(i32, i32, usize, usize)> = vec![];
        for ((x, y), dir) in self.bl_stables.iter(){
            let max_r = if load.f {6} else {2};
            for r in 0..max_r{
                kouho.push((*x, *y, *dir, r));
            }
        }
        kouho.shuffle(&mut rng);
        for (x, y, dir, r) in kouho{
            let load_ = load.rotated(r);
            if let Some(z) = self.get_z((x, y), dir, (load_.w, load_.h)){
                let pl = PlacedLoad::new(&load, (x, y, z, r), false, dir);
                if pl.can_place(&self.placedloads){
                    let mut field = self.clone();
                    field.add_placedload(pl);
                    ret.push(field);
                }
            }
            if ret.len()>=n{
                break;
            }
        }
        ret
    }

    fn get_z(&self, xy: (i32, i32), dir: usize, wh: (i32, i32)) -> Option<i32>{
        let (x, y) = xy;
        let (w, h) = wh;
        let dummy_pl = PlacedLoad::new(&Load{w, h, d: 1000, idx: 1004, f: false, g:false}, (x, y, 0, 0), true, dir);
        let mut min_z = 0;
        for pl in self.placedloads.iter(){
            let common_info = dummy_pl.common_info(pl);
            if common_info==8 && !pl.load.g{
                return None
            }
            if common_info==8 && min_z<pl.ze{
                min_z = pl.ze
            }
        }
        Some(min_z)
    }

    fn add_placedload(&mut self, pl: PlacedLoad){
        for pl2_ in self.placedloads.iter(){
            let common_info = pl.common_info(pl2_);
            /// 0: (xs, ys), 1(xs, ye), 2(xe, ys), 3 (xe, ye)
            match common_info{
                0 | 1 => {
                    let pl1 = if common_info%2==0 {&pl} else {pl2_};
                    let pl2 = if common_info%2==0 {pl2_} else {&pl};
                    let x = pl1.xe;
                    if pl1.ys<pl2.ys{
                        self.bl_stables.push(((x, pl2.ys), 1))
                    }
                    if pl1.ys>pl2.ys{
                        self.bl_stables.push(((x, pl1.ys), 3))
                    }
                    if pl1.ye>pl2.ye{
                        self.bl_stables.push(((x, pl2.ye), 0))
                    }
                    if pl1.ye<pl2.ye{
                        self.bl_stables.push(((x, pl1.ye), 2))
                    }
                }
                2 | 3 => {
                    let pl1 = if common_info%2==0 {&pl} else {pl2_};
                    let pl2 = if common_info%2==0 {pl2_} else {&pl};
                    let y = pl1.ye;
                    if pl1.xs<pl2.xs{
                        self.bl_stables.push(((pl2.xs, y), 2));
                    }
                    if pl1.xs>pl2.xs{
                        self.bl_stables.push(((pl1.xs, y), 3));
                    }
                    if pl1.xe>pl2.xe{
                        self.bl_stables.push(((pl2.xe, y), 0));
                    }
                    if pl1.xe<pl2.xe{
                        self.bl_stables.push(((pl1.xe, y), 1));
                    }
                },
                _ => {}
            }
        }
        // TODO: bl_stableをfilter
        if pl.load.idx<self.rest_load.len(){
            self.rest_load[pl.load.idx] -= 1;
        }
        self.placedloads.push(pl);
        self.score = None;
        self.get_score();
    }

    fn get_bl_stables(&self, input: &Input) -> Vec<((i32, i32), usize)>{
        let mut ret: Vec<((i32, i32), usize)> = vec![];
        // 0: (xs, ys), 1(xs, ye), 2(xe, ys), 3 (xe, ye)
        // 0: xで接触(self<oppo) 1: xで接触(self>oppo) 2: yで接触(self<oppo) 3: yで接触(self>oppo) 4: zで接触(self<oppo) 5: zで接触(self>oppo) 6: 接点なし だが上下にある (self<oppo) 7: 接点なし だが上下にある (self>oppo) 8: 共有領域あり 9: 接点なし(その他)
        for pl1_ in self.placedloads.iter(){
            for pl2_ in self.placedloads.iter(){
                let common_info = pl1_.common_info(pl2_);
                match common_info{
                    0 | 1 => {
                        let pl1 = if common_info%2==0 {pl1_} else {pl2_};
                        let pl2 = if common_info%2==0 {pl2_} else {pl1_};
                        let x = pl1.xe;
                        if pl1.ys<pl2.ys{
                            ret.push(((x, pl2.ys), 1))
                        }
                        if pl1.ys>pl2.ys{
                            ret.push(((x, pl1.ys), 3))
                        }
                        if pl1.ye>pl2.ye{
                            ret.push(((x, pl2.ye), 0))
                        }
                        if pl1.ye<pl2.ye{
                            ret.push(((x, pl1.ye), 2))
                        }
                    }
                    2 | 3 => {
                        let pl1 = if common_info%2==0 {pl1_} else {pl2_};
                        let pl2 = if common_info%2==0 {pl2_} else {pl1_};
                        let y = pl1.ye;
                        if pl1.xs<pl2.xs{
                            ret.push(((pl2.xs, y), 2));
                        }
                        if pl1.xs>pl2.xs{
                            ret.push(((pl1.xs, y), 3));
                        }
                        if pl1.xe>pl2.xe{
                            ret.push(((pl2.xe, y), 0));
                        }
                        if pl1.xe<pl2.xe{
                            ret.push(((pl1.xe, y), 1));
                        }
                    },
                    _ => {}
                }
            }
        }
        return ret
    }

    fn can_place(&self, input: &Input, idx: usize, xyzr: (i32, i32, i32, usize), dir: usize) -> bool{
        let pl1 = PlacedLoad::new(&input.loads[idx], xyzr, false, dir);
        pl1.can_place( &self.placedloads)
    }

    fn get_score(&mut self) -> usize{
        if let Some(score) = self.score{
            return score;
        }
        if self.placedloads.len()==self.N{
            let score =  self.get_raw_score();
            self.score = Some(score);
            return score;
        } else {
            let score =  self.get_mid_score();
            self.score = Some(score);
            return score;
        }
    }

    fn get_mid_score(&self) -> usize{
        let mut max_height = 0;
        for pl in self.placedloads.iter(){
            if pl.is_dummy{
                continue;
            }
            if max_height < pl.ze{
                max_height = pl.ze
            }
        }
        let order = self.get_order();
        let order_idx: Vec<usize> = order.iter().map(|idx| self.placedloads[*idx].load.idx).collect();
        let inv_cnt = inversion(&order_idx);
        let mut score = 1000+inv_cnt*1000+{if (max_height>self.D){100000000000} else {0}};
        score
    }

    fn get_raw_score(&self) -> usize{
        let mut max_height = 0;
        for pl in self.placedloads.iter(){
            if pl.is_dummy{
                continue;
            }
            if max_height < pl.ze{
                max_height = pl.ze
            }
        }
        let order = self.get_order();
        let order_idx: Vec<usize> = order.iter().map(|idx| self.placedloads[*idx].load.idx).collect();
        let inv_cnt = inversion(&order_idx);
        let mut score = 1000+(max_height as usize)+inv_cnt*1000+{if (max_height>self.D){100000000000} else {0}};
        score
    }


    fn get_order(&self) -> Vec<usize>{
        // トポソっぽくしながら一番idx低いの貪欲
        let mut ret: Vec<usize> = vec![];
        let mut depends: Vec<HashSet<usize>> = vec![HashSet::new(); self.placedloads.len()];
        let mut depends_rev: Vec<HashSet<usize>> = vec![HashSet::new(); self.placedloads.len()];
        let mut no_depends = BinaryHeap::new();
        for (i1, pl1) in self.placedloads.iter().enumerate(){
            if pl1.is_dummy{
                continue;
            }
            // 0: xで接触(self<oppo) 1: xで接触(self>oppo) 2: yで接触(self<oppo) 3: yで接触(self>oppo) 4: zで接触(self<oppo) 5: zで接触(self>oppo) 6: 接点なし だが上下にある (self<oppo) 7: 接点なし だが上下にある (self>oppo) 8: 共有領域あり 9: 接点なし(その他)
            for (i2, pl2) in self.placedloads.iter().enumerate(){
                if pl2.is_dummy{
                    continue;
                }
                let common_info = pl1.common_info(pl2);
                match common_info{
                    4|6 => {depends[i2].insert(i1);depends_rev[i1].insert(i2);},
                    _ => {}
                }
            }
        }
        for (i, depend) in depends.iter().enumerate(){
            if depend.is_empty(){
                no_depends.push((Reverse(self.placedloads[i].load.idx), i));
            }
        }
        while let Some((Reverse(idx), i)) = no_depends.pop(){
            if idx>=1000{
                continue;
            }
            ret.push(i);
            for i2 in depends_rev[i].iter(){
                depends[*i2].remove(&i);
                if depends[*i2].is_empty(){
                    no_depends.push((Reverse(self.placedloads[*i2].load.idx), *i2));
                }
            }
        }
        ret
    }
}

impl Ord for Field{
    fn cmp(&self, other: &Field) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

impl PartialOrd for Field {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.score.cmp(&other.score))
    }
}

impl PartialEq for Field {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Field{

}

impl Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ans: Vec<String> = vec![];
        for i in self.get_order(){
            let pl = &self.placedloads[i];
            ans.push(format!("{} {} {} {} {}", pl.load.idx, pl.r, pl.xs, pl.ys, pl.zs));
        }
        write!(f, "{}", ans.join("\n"))?;
        Ok(())
    }
}

fn beam_search(input: &Input, order: &Vec<usize>, beam_width: usize) -> Vec<Field>{
    let mut fields_base = vec![Field::new(input)];
    let mut fields_new: Vec<Field> = vec![];
    for (i, idx) in order.iter().enumerate(){
        let load = &input.loads[*idx];
        for field in fields_base.iter(){
            for neighbor in field.get_neighbor(input, load, 100){
                fields_new.push(neighbor);
            }
        }
        fields_new.sort_by_cached_key(|x| x.score);
        fields_base = fields_new.into_iter().take(beam_width).collect();
        fields_new = vec![];
        // println!("{}", fields_base[0]);
        // println!("--------------");
    }
    fields_base
}

fn chokudai_search(input: &Input, timer: &Instant, tl: f64, beam_width: usize) -> Field{
    let mut field_init = Field::new(input);
    let mut heaps = vec![BinaryHeap::new(); input.loads.len()+1];
    heaps[0].push(Reverse(field_init));
    while timer.elapsed().as_secs_f64()<tl{
        for t in 0..input.loads.len(){
            for i in 0..beam_width{
                if let Some(Reverse(field)) = heaps[t].pop(){
                    for field_next in field.get_all_neighbor(input){
                        heaps[t+1].push(Reverse(field_next));
                    }
                } else{
                    break;
                }
            }
        }
    }
    let Reverse(best_field) = heaps[input.loads.len()].pop().unwrap();
    best_field
}



#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    ans: Vec<usize>
}

impl State{
    fn get_neighbor(&self, input: &Input) -> (usize, usize){
        (1, 1)
    }
    fn update(&mut self, input: &Input, params: (usize, usize)){
        1+1;
    }
    fn undo(&mut self, input: &Input, params: (usize, usize)){
        1+1;
    }
    fn get_score(&self) -> usize{
        0
    }
}


impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dst: Vec<String> = self.ans.iter().map(|x| (x+1).to_string()).collect();
        write!(f, "{}", dst.join(" "))?;
        Ok(())
    }
}


fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer);
}

fn solve(input: &Input, timer:&Instant){
    let mut order = vec![];
    for (i, load) in input.loads.iter().enumerate(){
        if load.g{
            order.push(i);
        }
    }
    for (i, load) in input.loads.iter().enumerate(){
        if !load.g{
            order.push(i);
        }
    }
    let best_fields = chokudai_search(input, &timer, 1.5, 1);
    println!("{}", best_fields);
    eprintln!("{}", timer.elapsed().as_secs_f64());
}
