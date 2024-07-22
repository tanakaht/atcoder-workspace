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
use std::cmp::Reverse;
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



#[derive(Debug, Clone, Copy)]
struct Point{
    x: usize,
    y: usize,
    is_broken: bool,
    p: usize,
}

impl Point{
    fn new(x: usize, y: usize) -> Self{
        Self {x, y, is_broken:false, p:0}
    }

    fn near_point(&self) -> Vec<(usize, usize)>{
        let mut ret: Vec<(usize, usize)> = vec![];
        for (x, y) in [(self.x+1, self.y), (self.x-1, self.y), (self.x, self.y+1), (self.x, self.y-1)]{
            if (0..N).contains(&x) && (0..N).contains(&y){
                ret.push((x, y));
            }
        }
        ret
    }

    fn estimate_p(&self)->usize{
        self.p+250
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    N: usize,
    W: usize,
    K: usize,
    C: usize,
    waters: Vec<(usize, usize)>,
    houses: Vec<(usize, usize)>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            N_: usize,
            W: usize,
            K: usize,
            C: usize,
            AB_: [[usize; 2]; W],
            CD_: [[usize; 2]; K]
        }
        let waters: Vec<(usize, usize)> = AB_.iter().map(|x| (x[0], x[1])).collect();
        let houses: Vec<(usize, usize)> = CD_.iter().map(|x| (x[0], x[1])).collect();
        Self { N: N_, W, K, C, waters, houses }
    }
}

struct Field {
    M: [[Point; N]; N],
    C: usize,
    waters: Vec<(usize, usize)>,
    houses: Vec<(usize, usize)>
}

impl Field {
    fn new(input: &Input) -> Self{
        let mut M: [[Point; N]; N] = [[Point::new(0, 0); N]; N];
        for i in 0..N{
            for j in 0..N{
                M[i][j].x = i;
                M[i][j].y = j;
            }
        }
        Self {M, C:input.C, waters: input.waters.clone(), houses: input.houses.clone()}
    }

    fn solve(&self, timer: &Instant){

    }

    fn process(&self, i: usize, j:usize, p:usize) -> bool{
        let mut point = self.M[i][j];
        if point.is_broken{
            eprintln!("({}, {}) is already broken!", i, j);
            return false;
        }
        println!("{} {} {}", i, j, p);
        input! {
            r: usize
        }
        point.p += p;
        if r==0{
        } else if r==1{
            point.is_broken = true;
        } else if r==2 {
            exit(0);
        } else {
            exit(1);
        }
        return true;
    }
}


fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    let field = Field::new(&input);
    field.solve(&timer);
}
