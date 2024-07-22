#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
use itertools::Itertools;
use proconio::{input, marker::Chars, source::line::LineSource};
use std::io::{stdin, stdout, BufReader, Write};
use std::fmt::Display;
use std::cmp::{Reverse, Ordering, max, min};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;
use ndarray::prelude::*;
use grid::{Coordinate, CoordinateDiff, Map2d, ADJACENTS};


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    l: usize,
    n: usize,
    s: usize,
    //a: [i32; n], // a is Vec<i32>, n-array.
    xy: Vec<Coordinate> // `a` is Vec<Vec<i32>>, (m, n)-matrix.
}

impl Input {
    fn read_input() -> Self {
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            l: usize,
            n: usize,
            s: usize
        }
        let mut xy = vec![];
        for _ in 0..n{
            input! {
                from &mut source,
                x: usize,
                y: usize
            }
            xy.push(Coordinate::new(x, y, l));
        }
        Self { l, n, s, xy }
    }
}

struct Arrangement{
    map: Map2d<usize>,
    x: usize,
    n_planed_meansure: usize,
    id_points: Vec<CoordinateDiff>,
    id_for_output_cell: Vec<usize>,
}

impl Arrangement{
    fn new(input: &Input) -> Self{
        let mut map = Map2d::<usize>::new(vec![0; input.l*input.l], input.l);
        let (id_point, id_for_output_cell) = Self::solve_id_points_and_id_for_output_cell(input);
        let key_point_templature = Self::solve_key_point_templature(input);
        let other_point_templature = Self::solve_other_point_templature(input, &key_point_templature);
        let mut x = 0;
        let mut n_planed_meansure = 0;
        let mut id_points = vec![];
        let mut id_for_output_cell = vec![];
        for (i, c) in input.xy.iter().enumerate() {
            let mut templature = key_point_templature[i].1;
            if other_point_templature[c] > templature {
                templature = other_point_templature[c];
            }
            if templature > x {
                x = templature;
            }
            if templature > 0 {
                n_planed_meansure += 1;
            }
            map[c] = templature;
            arrangement[c] = i;
            id_points.push(CoordinateDiff::new(c.x, c.y));
            id_for_output_cell.push(i);
        }
        id_points.sort_by_key(|c| c.x);
        id_for_output_cell.sort_by_key(|i| input.xy[*i].x);
        Self { map, x, n_planed_meansure, id_points, id_for_output_cell }
    }


    fn solve_id_points_and_id_for_output_cell(&mut self, input: &Input) -> (Vec<Coordinate>, Vec<usize>){
        let mut id_points = vec![];
        let mut id_for_output_cell = vec![];
        for (i, c) in input.xy.iter().enumerate() {
            id_points.push(CoordinateDiff::new(c.x, c.y));
            id_for_output_cell.push(i);
        }
        id_points.sort_by_key(|c| c.x);
        id_for_output_cell.sort_by_key(|i| input.xy[*i].x);
        self.id_points = id_points;
        self.id_for_output_cell = id_for_output_cell;
    }

    fn solve_key_point_templature(input: &Input, id_point: &Vec<CoordinateDiff>, id_for_output_cell: &Vec<usize>) -> Vec<(Coordinate, usize)>{
        let mut key_point_templature = vec![];
        for (i, c) in input.xy.iter().enumerate() {
            key_point_templature.push((*c, 1000*i/input.n));
        }
        key_point_templature
    }
}

fn solve_key_point_templature(input: &Input) -> Vec<(Coordinate, usize)>{
    let mut key_point_templature = vec![];
    for (i, c) in input.xy.iter().enumerate() {
        key_point_templature.push((*c, 1000*i/input.n));
    }
    key_point_templature
}

fn solve_other_point_templature(input: &Input, key_point_templature: &Vec<(Coordinate, usize)>) -> Map2d<usize>{
    let mut map = Map2d::<usize>::new(vec![0; input.l*input.l], input.l);
    let key_points: HashSet<Coordinate> = key_point_templature.iter().map(|(c, _)| *c).collect();
    for (c, templature) in key_point_templature.iter() {
        map[c] = *templature;
    }
    for _ in 0..100{
        for x in 0..input.n{
            for y in 0..input.n{
                let c = Coordinate::new(x, y, input.l);
                if key_points.contains(&c) {continue;}
                let mut mean_neighbor_xy = c.get_adj().iter().map(|c| map[c]).sum::<usize>()/4;
                map[c] = mean_neighbor_xy;
            }
        }
    }
    map
}

fn solve_arrangement(input: &Input) -> Map2d<usize>{
    let key_point_templature = solve_key_point_templature(input);
    let map = solve_other_point_templature(input, &key_point_templature);
    map
}





fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    // get arrangement
    let arrangement = solve_arrangement(&input);
    // print arrangement by input.l
    for x in 0..input.l{
        for y in 0..input.l{
            print!("{} ", arrangement[Coordinate::new(x, y, input.l)]);
        }
        println!();
    }
    // 計測, 回答
    solve(&input, &timer, &arrangement, 3.8);
}

fn solve(input: &Input, timer:&Instant, arrangement: &Map2d<usize>, tl: f64){
    let hole_templature: Vec<usize> = input.xy.iter().map(|c| arrangement[*c]).collect();
    let mut measured_templature: Vec<Vec<usize>> = vec![vec![]; input.n];
    let stdin = std::io::stdin();
    for _ in 0..100{
        for i in 0..input.n{
            println!("{} {} {}", i, 0, 0);
            let mut source = LineSource::new(BufReader::new(stdin.lock()));
            input!{
                from &mut source,
                temp: usize
            }
            measured_templature[i].push(temp);
        }
    }
    println!("-1 -1 -1");
    let mean_templature: Vec<usize> = measured_templature.iter().map(|t| t.iter().sum::<usize>()/t.len()).collect();
    for (_, t) in mean_templature.iter().enumerate(){
        let mut best_i = 0;
        let mut best_diff = 1000000000;
        for (i, t2) in hole_templature.iter().enumerate(){
            let diff = (*t as i64- *t2 as i64).abs() as usize;
            if diff < best_diff{
                best_diff = diff;
                best_i = i;
            }
        }
        println!("{}", best_i);
    }
}


//借用して改変
#[allow(dead_code)]
mod grid {
    use itertools::Itertools;
    use std::cmp::min;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Coordinate {
        pub row: usize,
        pub col: usize,
        pub size: usize
    }

    impl Coordinate {
        pub const fn new(row: usize, col: usize, size: usize) -> Self {
            // ガバだけど, diffが100＊sizee以下ならOKなはず
            Self { row: (row.wrapping_add(100*size))%size, col: (col.wrapping_add(100*size))%size, size }
        }

        pub const fn to_index(&self) -> usize {
            self.row * self.size + self.col
        }

        pub fn dist(&self, other: &Self) -> usize {
            self.dist_1d(self.row, other.row) + self.dist_1d(self.col, other.col)
        }

        pub fn get_adj(&self) -> Vec<Coordinate> {
            ADJACENTS.iter().map(|&diff| *self + diff).collect()
        }

        fn dist_1d(&self, x0: usize, x1: usize) -> usize {
            min::<usize>((x0 as i64 - x1 as i64).abs() as usize, self.size-(x0 as i64 - x1 as i64).abs() as usize)
        }

    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CoordinateDiff {
        pub dr: usize,
        pub dc: usize,
    }

    impl CoordinateDiff {
        pub const fn new(dr: usize, dc: usize) -> Self {
            Self { dr, dc }
        }

        pub const fn invert(&self) -> Self {
            Self::new(0usize.wrapping_sub(self.dr), 0usize.wrapping_sub(self.dc))
        }
    }

    impl std::ops::Add<CoordinateDiff> for Coordinate {
        type Output = Coordinate;

        fn add(self, rhs: CoordinateDiff) -> Self::Output {
            Coordinate::new(self.row.wrapping_add(rhs.dr), self.col.wrapping_add(rhs.dc), self.size)
        }
    }

    pub const ADJACENTS: [CoordinateDiff; 4] = [
        CoordinateDiff::new(!0, 0),
        CoordinateDiff::new(0, 1),
        CoordinateDiff::new(1, 0),
        CoordinateDiff::new(0, !0),
    ];

    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        pub width: usize,
        pub height: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, width: usize) -> Self {
            let height = map.len() / width;
            debug_assert!(width * height == map.len());
            Self { width, height, map }
        }
    }

    impl<T> std::ops::Index<Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coordinate) -> &Self::Output {
            &self.map[coordinate.to_index()]
        }
    }

    impl<T> std::ops::IndexMut<Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.to_index()]
        }
    }

    impl<T> std::ops::Index<&Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coordinate) -> &Self::Output {
            &self.map[coordinate.to_index()]
        }
    }

    impl<T> std::ops::IndexMut<&Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.to_index()]
        }
    }

    impl<T> std::ops::Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &self.map[begin..end]
        }
    }

    impl<T> std::ops::IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &mut self.map[begin..end]
        }
    }
}
