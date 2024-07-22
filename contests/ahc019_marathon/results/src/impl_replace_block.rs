#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
use itertools::Itertools;
use num::{range, ToPrimitive};
use proconio::{input, marker::Chars};
use rand_core::block;
use std::iter::FromIterator;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fmt::Display;
use std::cmp::{Reverse, Ordering};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;
use ndarray::prelude::*;
use num::BigUint;

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



// fn rotate(X: &Array3<u16>, axis: (usize, usize), k: usize) -> Array3<u16> {
//     let mut Y = X.clone();
//     for _ in 0..k {
//         Y = Y.reversed_axes();
//         Y.swap_axes(axis.0, axis.1);
//         //Y = Y.reversed_axes();
//     }
//     Y
// }

fn rotate_x(a: &Array3<u16>, k: usize) -> Array3<u16> {
    let mut b = a.clone();
    for _ in 0..k {
        b = b.slice(s![.., .., ..;-1]).to_owned();
        b.swap_axes(1, 2);
    }
    b
}

fn rotate_y(a: &Array3<u16>, k: usize) -> Array3<u16> {
    let mut b = a.clone();
    for _ in 0..k {
        b = b.slice(s![.., .., ..;-1]).to_owned();
        b.swap_axes(0, 2);
    }
    b
}


fn pad_axis(a: &Array3<u16>, pad: ((usize, usize), (usize, usize), (usize, usize))) -> Array3<u16> {
    let old_shape = a.shape();
    let pad_shape = (old_shape[0]+(pad.0).0+(pad.0).1, old_shape[1]+(pad.1).0+(pad.1).1, old_shape[2]+(pad.2).0+(pad.2).1);
    let mut padded = Array3::zeros(pad_shape);
    for x in 0..old_shape[0]{
        for y in 0..old_shape[1]{
            for z in 0..old_shape[2]{
                padded[[x+(pad.0).0, y+(pad.1).0, z+(pad.2).0]] = a[[x, y, z]]
            }
        }
    }
    padded
}

fn get_place_from_dir(xyz: (usize, usize, usize), dir: usize, D: usize) -> Option<(usize, usize, usize)>{
    let (x, y, z) = xyz;
    // if min!(x, y, z)==0{
    //     return None;
    // }
    let (x_, y_, z_) = match dir{
        0 => (x, y, z+1),
        1 => (x, y, z-1),
        2 => (x, y+1, z),
        3 => (x, y-1, z),
        4 => (x+1, y, z),
        5 => (x-1, y, z),
        _ => (x, y, z)
    };
    if max!(x_, y_, z_)<D{
        Some((x_, y_, z_))
    } else {
        None
    }
}

fn get_place_from_dir2(xyz: (usize, usize, usize), dir: usize, XYZ: (usize, usize, usize)) -> Option<(usize, usize, usize)>{
    let (x, y, z) = xyz;
    if min!(x, y, z)==0{
        return None;
    }
    let (x_, y_, z_) = match dir{
        0 => (x, y, z+1),
        1 => (x, y, z-1),
        2 => (x, y+1, z),
        3 => (x, y-1, z),
        4 => (x+1, y, z),
        5 => (x-1, y, z),
        _ => (x, y, z)
    };
    if x_<XYZ.0 && y_<XYZ.1 && z_<XYZ.2{
        Some((x_, y_, z_))
    } else {
        None
    }
}

fn places2dir(xyz: (usize, usize, usize), xyz2: (usize, usize, usize)) -> usize{
    let (x, y, z) = xyz;
    let (x_, y_, z_) = xyz2;
    let diff = ((x_ as i32)-(x as i32), (y_ as i32)-(y as i32), (z_ as i32)-(z as i32));
    match diff{
        (0, 0, 1) => 0,
        (0, 0, -1) => 1,
        (0, 1, 0) => 2,
        (0, -1, 0) => 3,
        (1, 0, 0) => 4,
        (-1, 0, 0) => 5,
        _ => 10000,
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Block {
    id: usize,
    serial: String,
    is_filled: Array3<u16>,
    place: (usize, usize, usize),
    n_cube: usize,
}

impl Block{
    fn single_cube(id: usize, place: (usize, usize, usize)) -> Self{
        let serial: String = String::from("1 1 1 1");
        Self{id, serial, is_filled: array![[[1]]], place, n_cube: 1}
    }

    fn get_bbox(&self)-> (usize, usize, usize, usize, usize, usize){
        let (xs, ys, zs) = self.place;
        let shape = self.is_filled.shape();
        let (xe, ye, ze) = (xs+shape[0]-1, ys+shape[1]-1, zs+shape[2]-1);
        (xs, xe, ys, ye, zs, ze)
    }

    fn has_block(&self, xyz: (usize, usize, usize)) -> bool{
        let (x, y, z) = xyz;
        let (xs, ys, zs) = self.place;
        let shape = self.is_filled.shape();
        let (mut xe, mut ye, mut ze) = (xs+shape[0]-1, ys+shape[1]-1, zs+shape[2]-1);
        if !(xs<=x && x<= xe && ys<=y && y<= ye && zs<=z && z<= ze){
            return false
        }
        self.is_filled[[x-xs, y-ys, z-zs]]==1
    }

    fn get_cubes(&self) -> Vec<(usize, usize, usize)>{
        let mut places = vec![];
        let (xs, ys, zs) = self.place;
        let shape = self.is_filled.shape();
        let (xe, ye, ze) = (xs+shape[0]-1, ys+shape[1]-1, zs+shape[2]-1);
        for x in xs..(xe+1){
            for y in ys..(ye+1){
                for z in zs..(ze+1){
                    if self.is_filled[[x-xs, y-ys, z-zs]]==1{
                        places.push((x, y, z));
                    }
                }
            }
        }
        places
    }

    fn cal_serial(&self) -> String{
        if !self.is_connected(){
            let mut rng = rand::thread_rng();
            // eprintln!("{:?}", self);
            return  format!("invalid_{}", rng.gen::<usize>());
        }
        let mut min_serial: Option<String> = None;
        for xr in 0..4{
            let mut X =rotate_x(&self.is_filled, xr);
            for yr in 0..4{
                let shape_str = X.shape().iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ");
                let iter_str = X.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ");
                let serial = format!("{} {}", shape_str, iter_str);
                if let Some(serial_) = &min_serial{
                    if serial < *serial_{
                        min_serial = Some(serial);
                    }
                } else {
                    min_serial = Some(serial);
                }
                if yr<3{
                    X = rotate_y(&X, 1);
                }
            }
        }
        min_serial.unwrap()
    }

    fn is_connected(&self) -> bool{
        let cubes = self.get_cubes();
        let mut place2id: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let shape = self.is_filled.shape();
        let XYZ = (shape[0], shape[1], shape[2]);
        for (i, xyz) in cubes.iter().enumerate(){
            place2id.insert(*xyz, i);
        }
        let mut uf = UnionFind::new(cubes.len());
        for (i, xyz) in cubes.iter().enumerate(){
            for dir in 0..6{
                if let Some(xyz_) = get_place_from_dir(*xyz, dir, 1000){ // Dが適当だがこれで良い
                    if let Some(j) = place2id.get(&xyz_){
                        uf.union(i, *j);
                    }
                }
            }
        }
        uf.size(0)==cubes.len()
    }

    fn update_serial(&mut self){
        self.serial = self.cal_serial();
    }

    fn update_cubes(&mut self, cubes: &Vec<(usize, (usize, usize, usize))>){
        let (mut xs, mut ys, mut zs) = self.place;
        let shape = self.is_filled.shape();
        let (mut xe, mut ye, mut ze) = (xs+shape[0]-1, ys+shape[1]-1, zs+shape[2]-1);
        let (min_x, min_y, min_z) = cubes.iter().fold((10000, 10000, 10000), |(min_x, min_y, min_z), &(flg, (x, y, z))| (min_x.min(x), min_y.min(y), min_z.min(z)));
        let (max_x, max_y, max_z) = cubes.iter().fold((0, 0, 0), |(max_x, max_y, max_z), &(flg, (x, y, z))| (max_x.max(x), max_y.max(y), max_z.max(z)));
        let pad = ((xs-min!(min_x, xs), max!(max_x, xe)-xe), (ys-min!(min_y, ys), max!(max_y, ye)-ye), (zs-min!(min_z, zs), max!(max_z, ze)-ze));
        if (pad.0).0+(pad.0).1+(pad.1).0+(pad.1).1+(pad.2).0+(pad.2).1>0{
            self.is_filled = pad_axis(&self.is_filled, pad);
            xs -= (pad.0).0;
            ys -= (pad.1).0;
            zs -= (pad.2).0;
            self.place = (xs, ys, zs);
        }
        // cube追加, 削除
        for (flg, (x, y, z)) in cubes.iter(){
            if *flg==0{
                assert!(self.is_filled[[*x-xs, *y-ys, *z-zs]] != 0);
                self.is_filled[[*x-xs, *y-ys, *z-zs]] = 0;
                self.n_cube -= 1;
            } else {
                assert!(self.is_filled[[*x-xs, *y-ys, *z-zs]] != 1);
                self.is_filled[[*x-xs, *y-ys, *z-zs]] = 1;
                self.n_cube += 1;
            }
        }
        // assert!(self.n_cube==self.is_filled.sum() as usize);
        if self.n_cube==0{
            self.serial = String::from("");
            self.is_filled = array![[[0]]];
            return;
        }
        //縮小判定
        while self.is_filled.slice(s![0, .., ..]).sum()==0{
            self.is_filled = self.is_filled.slice(s![1..,..,..]).to_owned();
            self.place.0 += 1;
        }
        while self.is_filled.slice(s![.., 0, ..]).sum()==0{
            self.is_filled = self.is_filled.slice(s![..,1..,..]).to_owned();
            self.place.1 += 1;
        }
        while self.is_filled.slice(s![.., .., 0]).sum()==0{
            self.is_filled = self.is_filled.slice(s![..,..,1..]).to_owned();
            self.place.2 += 1;
        }
        while self.is_filled.slice(s![-1, .., ..]).sum()==0{
            self.is_filled = self.is_filled.slice(s![..-1,..,..]).to_owned();
        }
        while self.is_filled.slice(s![.., -1, ..]).sum()==0{
            self.is_filled = self.is_filled.slice(s![..,..-1,..]).to_owned();
        }
        while self.is_filled.slice(s![.., .., -1]).sum()==0{
            self.is_filled = self.is_filled.slice(s![..,..,..-1]).to_owned();
        }
        // serial更新
        self.update_serial();
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    D: usize,
    f1: Array2<u16>,
    r1: Array2<u16>,
    f2: Array2<u16>,
    r2: Array2<u16>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            D: usize,
            f1_: [Chars; D],
            r1_: [Chars; D],
            f2_: [Chars; D],
            r2_: [Chars; D],
        }
        let mut f1: Array2<u16> = Array2::zeros((D, D));
        let mut r1: Array2<u16> = Array2::zeros((D, D));
        let mut f2: Array2<u16> = Array2::zeros((D, D));
        let mut r2: Array2<u16> = Array2::zeros((D, D));
        for (base, target) in [(f1_, &mut f1), (r1_, &mut r1), (f2_, &mut f2), (r2_, &mut r2)].iter_mut(){
            for i in 0..D{
                for (j, flg) in base[i].iter().enumerate(){
                    if *flg=='1'{
                        target[[i, j]] = 1;
                    }
                }
            }
        }
        Self { D, f1, r1, f2, r2}
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Params{
    x: usize
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Field{
    D: usize,
    f: Array2<u16>,
    r: Array2<u16>,
    filled_block_id: Array3<u16>,
    can_place: Array3<u16>,
    blocks: HashMap<usize, Block>,
    adj: HashMap<String, HashMap<String, HashSet<((usize, usize, usize), (usize, usize, usize))>>>,
    pre_cmd: Vec<(usize, (usize, usize, usize))>,
    max_place: usize,
    none_block_serial: String,

}

// TODO: adjをできたらやる
impl Field{
    fn new(input: &Input, f: &Array2<u16>, r: &Array2<u16>) -> Self{
        let mut filled_block_id: Array3<u16> = Array3::zeros((input.D, input.D, input.D));
        let mut can_place: Array3<u16> = Array3::zeros((input.D, input.D, input.D));
        let mut blocks: HashMap<usize, Block> = HashMap::new();
        let mut max_place: usize = 0;
        // serial -> (serial -> id1, cubeplace1, id2, cubeplace2)
        let mut adj: HashMap<String, HashMap<String, HashSet<((usize, usize, usize), (usize, usize, usize))>>> = HashMap::new();
        for x in 0..input.D{
            for y in 0..input.D{
                for z in 0..input.D{
                    if (f[[z, x]]==1_u16) & (r[[z, y]]==1_u16){
                        can_place[[x, y, z]] = 1;
                        max_place += 1;
                    }
                }
            }
        }
        Self {D: input.D, f: f.clone(), r: r.clone(), filled_block_id, can_place, blocks, adj, pre_cmd: vec![], max_place, none_block_serial: String::from("")}
    }

    fn get_serial(&self, xyz: (usize, usize, usize)) -> &String{
        let (x, y, z) = xyz;
        let block_id = self.filled_block_id[[x, y, z]] as usize;
        if block_id==0{
            return &self.none_block_serial
        }
        &self.blocks[&block_id].serial
    }

    fn get_serials(&self, xyzs: &Vec<(usize, usize, usize)>) -> Vec<&String>{
        let mut serials = vec![];
        for xyz in xyzs.iter(){
            let serial = self.get_serial(*xyz);
            serials.push(serial);
        }
        serials.sort();
        serials
    }

    fn random_init(&mut self, n: usize) -> bool{
        // まず、f, rが1なところに111のcubeを追加、キューブ数がn以下なら適当にcube配置
        // let mut cnt = 0;
        // for x in 0..self.D{
        //     for y in 0..self.D{
        //         for z in 0..self.D{
        //             if (self.f[[z, x]]==1_u16) & (self.r[[z, y]]==1_u16){
        //                 let new_block = Block::single_cube(self.get_new_block_id(), (x, y, z));
        //                 self.add_block(new_block);
        //                 cnt += 1;
        //             }
        //         }
        //     }
        // }
        self.blocks = HashMap::new();
        self.filled_block_id = Array3::zeros((self.D, self.D, self.D));
        self.adj = HashMap::new();
        let mut rng = rand::thread_rng();
        let mut cnt = 0;
        let mut silhouet_cnt = self.f.sum()+self.r.sum();
        while cnt<n && silhouet_cnt>0{
            let (x, y, z) = (rng.gen::<usize>()%self.D, rng.gen::<usize>()%self.D, rng.gen::<usize>()%self.D);
            if (self.f[[z, x]]==1_u16) & (self.r[[z, y]]==1_u16) & (self.filled_block_id[[x, y, z]]==0_u16){
                let silhouet_diff = (self.filled_block_id.slice(s![x, .., z]).sum()==0_u16) as u16+(self.filled_block_id.slice(s![.., y, z]).sum()==0_u16) as u16;
                if (silhouet_diff>0){
                    let new_block = Block::single_cube(self.get_new_block_id(), (x, y, z));
                    self.add_block(new_block);
                    cnt += 1;
                    silhouet_cnt -= silhouet_diff;
                }
            }
        }
        while cnt<n{
            let (x, y, z) = (rng.gen::<usize>()%self.D, rng.gen::<usize>()%self.D, rng.gen::<usize>()%self.D);
            if (self.f[[z, x]]==1_u16) & (self.r[[z, y]]==1_u16) & (self.filled_block_id[[x, y, z]]==0_u16){
                let new_block = Block::single_cube(self.get_new_block_id(), (x, y, z));
                self.add_block(new_block);
                cnt += 1;
            }
        }
        self.is_ok()
    }

    fn get_new_block_id(&self) -> usize{
        for id in 1..5000{
            if !self.blocks.contains_key(&id){
                return id
            }
        }
        5000
    }

    fn update_adj_rm(&mut self, cubes: &HashSet<(usize, usize, usize)>){
        for xyz in cubes.iter(){
            for dir in 0..6{
                if let Some(xyz_) = get_place_from_dir(*xyz, dir, self.D){
                    if self.filled_block_id[*xyz]==self.filled_block_id[xyz_] || self.can_place[xyz_]==0{
                        continue;
                    }
                    let serial = self.get_serial(*xyz).clone();
                    let serial_ = self.get_serial(xyz_).clone();
                    if serial==self.none_block_serial && serial_==self.none_block_serial{
                        continue;
                    }
                    if let Some(adj) = self.adj.get_mut(&serial){
                        if let Some(adj_) = adj.get_mut(&serial_){
                            adj_.remove(&(*xyz, xyz_));
                            if adj_.is_empty(){
                                adj.remove(&serial_);
                            }
                        }
                        if adj.is_empty(){
                            self.adj.remove(&serial);
                        }
                    }
                    if let Some(adj) = self.adj.get_mut(&serial_){
                        if let Some(adj_) = adj.get_mut(&serial){
                            adj_.remove(&(xyz_, *xyz));
                            if adj_.is_empty(){
                                adj.remove(&serial);
                            }
                        }
                        if adj.is_empty(){
                            self.adj.remove(&serial_);
                        }
                    }
                    // self.adj.get_mut(&serial).unwrap().get_mut(&serial_).unwrap().remove(&(*xyz, xyz_));
                    // self.adj.get_mut(&serial_).unwrap().get_mut(&serial).unwrap().remove(&(xyz_, *xyz));
                    // if self.adj.get_mut(&serial).unwrap().get_mut(&serial_).unwrap().is_empty(){
                    //     self.adj.get_mut(&serial).unwrap().remove(&serial_);
                    //     if self.adj.get_mut(&serial).unwrap().is_empty(){
                    //         self.adj.remove(&serial);
                    //     }
                    // }
                    // if self.adj.get_mut(&serial_).unwrap().get_mut(&serial).unwrap().is_empty(){
                    //     self.adj.get_mut(&serial_).unwrap().remove(&serial);
                    //     if self.adj.get_mut(&serial_).unwrap().is_empty(){
                    //         self.adj.remove(&serial_);
                    //     }
                    // }
                }
            }
        }
    }

    fn update_adj_add(&mut self, cubes: &HashSet<(usize, usize, usize)>){
        for xyz in cubes.iter(){
            for dir in 0..6{
                if let Some(xyz_) = get_place_from_dir(*xyz, dir, self.D){
                    if self.filled_block_id[*xyz]==self.filled_block_id[xyz_] || self.can_place[xyz_]==0{
                        continue;
                    }
                    let serial = self.get_serial(*xyz).clone();
                    let serial_ = self.get_serial(xyz_).clone();
                    if serial==self.none_block_serial && serial_==self.none_block_serial{
                        continue;
                    }
                    self.adj.entry(serial.clone()).or_insert(HashMap::new()).entry(serial_.clone()).or_insert(HashSet::new()).insert((*xyz, xyz_));
                    self.adj.entry(serial_.clone()).or_insert(HashMap::new()).entry(serial.clone()).or_insert(HashSet::new()).insert((xyz_, *xyz));
            }
            }
        }
    }

    fn add_block(&mut self, block: Block){
        let id = self.get_new_block_id();
        let (xs, ys, zs) = block.place;
        let shape = block.is_filled.shape();
        let (xe, ye, ze) = (xs+shape[0]-1, ys+shape[1]-1, zs+shape[2]-1);
        let updated_cubes = HashSet::from_iter(block.get_cubes());
        self.update_adj_rm(&updated_cubes);
        for xyz in updated_cubes.iter(){
            self.filled_block_id[*xyz] = id as u16;
        }
        self.blocks.insert(id, block);
        self.update_adj_add(&updated_cubes);
    }

    fn update(&mut self, id_xyzs: &Vec<(usize, (usize, usize, usize))>) -> (Vec<String>, Vec<String>){
        // xyzのブロックをid_xに変更し、before, afterのserialのvecを返す
        let mut change_block_info: HashMap<usize, Vec<(usize, (usize, usize, usize))>> = HashMap::new();
        let mut updated_cubes = HashSet::new();
        self.pre_cmd = vec![];
        // 変更箇所を集計
        for (new_block_id, xyz) in id_xyzs{
            let (x, y, z) = xyz;
            let old_block_id = self.filled_block_id[[*x, *y, *z]] as usize;
            self.pre_cmd.push((old_block_id, (*x, *y, *z)));
            (change_block_info.entry(old_block_id).or_insert(vec![])).push((0, (*x, *y, *z)));
            (change_block_info.entry(*new_block_id).or_insert(vec![])).push((1, (*x, *y, *z)));
        }
        for (block_id, cubes) in change_block_info.iter(){
            if *block_id==0{
                for (_, xyz) in cubes.iter(){
                    updated_cubes.insert(*xyz);
                }
            } else if let Some(block) = self.blocks.get(block_id) {
                for xyz in block.get_cubes(){
                    updated_cubes.insert(xyz);
                }
            }
        }
        // 変更前の情報を記録等
        let mut old_serials: Vec<String> = vec![];
        let mut new_serials: Vec<String> = vec![];
        for block_id in change_block_info.keys(){
            if let Some(block) = self.blocks.get(block_id){
                old_serials.push(block.serial.clone());
            }
        }
        // let mut updated_cubes = HashSet::new();
        // TODO: updated_cubesがあっているか再確認
        // for x in 0..(self.D*self.D*self.D){
        //     let (x, y, z)= (x/(self.D*self.D), (x/self.D)%self.D, x%self.D);
        //     if self.can_place[[x, y, z]]==1_u16{
        //         updated_cubes.insert((x, y, z));
        //     }
        // }
        self.update_adj_rm(&updated_cubes);
        // block変更
        for (block_id, cubes) in change_block_info.iter_mut(){
            if let  Some(block) = self.blocks.get_mut(block_id){
                for (flg, xyz) in cubes.iter(){
                    if *flg==1{
                        self.filled_block_id[*xyz] = (*block_id) as u16;
                    }
                }
                block.update_cubes(cubes);
            } else if *block_id!=0{
                let (flg, xyz) = cubes.pop().unwrap();
                assert!(flg==1, "flg should be 1");
                let mut new_block = Block::single_cube(*block_id, xyz);
                self.filled_block_id[xyz] = (*block_id) as u16;
                for (flg, xyz) in cubes.iter(){
                    if *flg==1{
                        self.filled_block_id[*xyz] = (*block_id) as u16;
                    }
                }
                new_block.update_cubes(cubes);
                self.blocks.insert(*block_id, new_block);
                //self.add_block(new_block);
            } else {
                for (flg, xyz) in cubes.iter(){
                    if *flg==1{
                        self.filled_block_id[*xyz] = (*block_id) as u16;
                    }
                }
            }
            if let Some(block) = self.blocks.get(block_id){
                if block.n_cube==0{
                    self.blocks.remove_entry(block_id);
                } else {
                    new_serials.push(block.serial.clone());
                }
            }
        }
        // adj修正
        self.update_adj_add(&updated_cubes);
        (old_serials, new_serials)
    }

    fn undo(&mut self){
        let pre_cmd = self.pre_cmd.clone();
        self.update(&pre_cmd);
        self.pre_cmd = vec![];
    }

    fn can_remove_block(&self, block_id: usize) -> bool{
        if let Some(block) = self.blocks.get(&block_id){
            for (x, y, z) in block.get_cubes(){
                let mut f_ok = false;
                let mut r_ok = false;
                for x_ in 0..self.D{
                    let block_id_ = self.filled_block_id[[x_, y, z]] as usize;
                    if block_id_!=0 && block_id_!=block_id{
                        f_ok = true;
                        break;
                    }
                }
                for y_ in 0..self.D{
                    let block_id_ = self.filled_block_id[[x, y_, z]] as usize;
                    if block_id_!=0 && block_id_!=block_id{
                        r_ok = true;
                        break;
                    }
                }
                if !(f_ok && r_ok){
                    return false
                }
            }
            true
        } else {
            false
        }
    }

    fn can_remove_cube(&self, xyz: (usize, usize, usize)) -> bool{
        let (x, y, z) = xyz;
        // let block_id = self.filled_block_id[[x, y, z]];
        // let mut f_ok = false;
        // let mut r_ok = false;
        // for x_ in 0..self.D{
        //     let block_id_ = self.filled_block_id[[x_, y, z]];
        //     if block_id_!=0 && block_id_!=block_id{
        //         f_ok = true;
        //         break;
        //     }
        // }
        // for y_ in 0..self.D{
        //     let block_id_ = self.filled_block_id[[x, y_, z]];
        //     if block_id_!=0 && block_id_!=block_id{
        //         r_ok = true;
        //         break;
        //     }
        // }
        // f_ok && r_ok
    (self.filled_block_id.slice(s![x, .., z]).sum()!=self.filled_block_id[[x, y, z]]) && (self.filled_block_id.slice(s![.., y, z]).sum()!=self.filled_block_id[[x,y,z]])
    }

    fn is_ok(&self) -> bool{
        let mut flg = true;
        for x in 0..self.D{
            for z in 0..self.D{
                if self.f[[z, x]]!=0{
                    flg &= (self.filled_block_id.slice(s![x, .., z]).sum()!=0);
                }
            }
        }
        for y in 0..self.D{
            for z in 0..self.D{
                if self.r[[z, y]]!=0{
                    flg &= (self.filled_block_id.slice(s![.., y, z]).sum()!=0);
                }
            }
        }
        flg
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    D: usize,
    field1: Field,
    field2: Field,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut field1: Field = Field::new(input, &input.f1, &input.r1);
        let mut field2: Field = Field::new(input, &input.f2, &input.r2);
        let mut n = min!(field1.max_place, field2.max_place)/10;
        loop{
            if field1.random_init(n)&field2.random_init(n){
                break;
            }
            // println!("{} {}", field1.is_ok(), field2.is_ok());
            n = min!(n+1, min!(field1.max_place, field2.max_place));
            //n = (n+min!(field1.max_place, field2.max_place))/2;
        }
        Self {D: input.D, field1, field2}
    }

    fn update_if_ok(&mut self, id_xyzs1: &Vec<(usize, (usize, usize, usize))>, id_xyzs2: &Vec<(usize, (usize, usize, usize))>) -> bool{
        if id_xyzs1.is_empty() && id_xyzs2.is_empty(){
            return false
        }
        let res1 = self.field1.update(id_xyzs1);
        let res2 = self.field2.update(id_xyzs2);
        // serial一致かチェック
        // TODO: 集合として
        let mut counter1: HashMap<String, i32> = HashMap::new();
        let mut counter2: HashMap<String, i32> = HashMap::new();
        for s in res1.0{
            *counter1.entry(s).or_insert(0) -= 1;
        }
        for s in res1.1{
            *counter1.entry(s).or_insert(0) += 1;
        }
        for s in res2.0{
            *counter2.entry(s).or_insert(0) -= 1;
        }
        for s in res2.1{
            *counter2.entry(s).or_insert(0) += 1;
        }
        for s in counter1.keys(){
            if counter1[s] != *counter2.get(s).unwrap_or(&0){
                self.undo();
                return false
            }
        }
        for s in counter2.keys(){
            if counter2[s] != *counter1.get(s).unwrap_or(&0){
                self.undo();
                return false
            }
        }
        true
    }

    fn undo(&mut self){
        self.field1.undo();
        self.field2.undo();
    }

    fn get_neighbor(&mut self) -> (Vec<(usize, (usize, usize, usize))>, Vec<(usize, (usize, usize, usize))>){
        let mut rng = rand::thread_rng();
        let mode_flg = rng.gen::<usize>()%100;
        if mode_flg < 30{
            // 近傍1: 隣のブロックに一つcube渡す
            for _ in 0..100{
                if let Some(serial) = self.field1.adj.keys().choose(&mut rng){
                    // if serial==&self.field1.none_block_serial{
                    //     continue;
                    // }
                    if let Some(serial_) = self.field1.adj.get(serial).unwrap().keys().choose(&mut rng){
                        if !(self.field2.adj.contains_key(serial) && self.field2.adj.get(serial).unwrap().contains_key(serial_)){
                            continue;
                        }
                        let (xyz1, xyz1_) = self.field1.adj.get(serial).unwrap().get(serial_).unwrap().iter().choose(&mut rng).unwrap();
                        let (xyz2, xyz2_) = self.field2.adj.get(serial).unwrap().get(serial_).unwrap().iter().choose(&mut rng).unwrap();
                        if serial==&self.field1.none_block_serial && (!self.field1.can_remove_cube(*xyz1_) || !self.field2.can_remove_cube(*xyz2_)){
                            continue;
                        }
                        return (vec![(self.field1.filled_block_id[*xyz1] as usize, *xyz1_)],
                                vec![(self.field2.filled_block_id[*xyz2] as usize, *xyz2_)])
                    }
                }
            }
            (vec![], vec![])
        } else if mode_flg < 60{
            // 近傍2: 隣のblock同士で結合
            for _ in 0..100{
                if let Some(serial) = self.field1.adj.keys().choose(&mut rng){
                    if serial==&self.field1.none_block_serial{
                        continue;
                    }
                    if let Some(serial_) = self.field1.adj.get(serial).unwrap().keys().choose(&mut rng){
                        if serial_==&self.field1.none_block_serial{
                            continue;
                        }
                            if !(self.field2.adj.contains_key(serial) && self.field2.adj.get(serial).unwrap().contains_key(serial_)){
                            continue;
                        }
                        let (xyz1, xyz1_) = self.field1.adj.get(serial).unwrap().get(serial_).unwrap().iter().choose(&mut rng).unwrap();
                        let (xyz2, xyz2_) = self.field2.adj.get(serial).unwrap().get(serial_).unwrap().iter().choose(&mut rng).unwrap();
                        let block_id_1 = self.field1.filled_block_id[*xyz1] as usize;
                        let block_id_2 = self.field2.filled_block_id[*xyz2] as usize;
                        let mut change_block_info_1 = vec![];
                        let mut change_block_info_2 = vec![];
                        // eprintln!("{} {} {} {} s1:{} s2:{}", block_id_1, self.field1.filled_block_id[*xyz1_], block_id_2, self.field2.filled_block_id[*xyz2_], serial, serial_);
                        // eprintln!("1, {}: {}", self.field1.filled_block_id[*xyz1_], serial_);
                        // eprintln!("2, {}: {}", self.field2.filled_block_id[*xyz2_], serial_);
                        // assert!(self.field1.blocks.get(&(self.field1.filled_block_id[*xyz1_] as usize)).is_some());
                        for xyz1__ in self.field1.blocks.get(&(self.field1.filled_block_id[*xyz1_] as usize)).unwrap().get_cubes(){
                            change_block_info_1.push((block_id_1, xyz1__));
                        }
                        for xyz2__ in self.field2.blocks.get(&(self.field2.filled_block_id[*xyz2_] as usize)).unwrap().get_cubes(){
                            change_block_info_2.push((block_id_2, xyz2__));
                        }
                        return (change_block_info_1, change_block_info_2)
                    }
                }
            }
            (vec![], vec![])
        } else  if mode_flg <= 70{
            // 近傍3: 消せるblockを削除
            let mut can_remove_block_id1 = vec![];
            let mut can_remove_block_id2 = vec![];
            for block_id in self.field1.blocks.keys(){
                if self.field1.can_remove_block(*block_id){
                    can_remove_block_id1.push(*block_id);
                }
            }
            for block_id in self.field2.blocks.keys(){
                if self.field2.can_remove_block(*block_id){
                    can_remove_block_id2.push(*block_id);
                }
            }
            can_remove_block_id1.shuffle(&mut rng);
            can_remove_block_id2.shuffle(&mut rng);
            for block_id1 in can_remove_block_id1.iter(){
                let block1 = self.field1.blocks.get(block_id1).unwrap();
                for block_id2 in can_remove_block_id2.iter(){
                    let block2 = self.field2.blocks.get(block_id2).unwrap();
                    if block1.serial==block2.serial{
                        let mut change_block_info_1 = vec![];
                        let mut change_block_info_2 = vec![];
                        // eprintln!("{} {} {} {} s1:{} s2:{}", block_id_1, self.field1.filled_block_id[*xyz1_], block_id_2, self.field2.filled_block_id[*xyz2_], serial, serial_);
                        for xyz1 in block1.get_cubes(){
                            change_block_info_1.push((0, xyz1));
                        }
                        for xyz2 in block2.get_cubes(){
                            change_block_info_2.push((0, xyz2));
                        }
                        return (change_block_info_1, change_block_info_2)
                    }
                }
            }
            (vec![], vec![])
        } else {
            // 近傍4: ブロックを移動
            let mut field_block_ids = vec![];
            for block_id in self.field1.blocks.keys(){
                field_block_ids.push((1, *block_id));
            }
            for block_id in self.field2.blocks.keys(){
                field_block_ids.push((2, *block_id));
            }
            field_block_ids.shuffle(&mut rng);
            for (field_id, block_id) in field_block_ids.iter(){
                let field = if (*field_id==1) {&self.field1} else {&self.field2};
                if !field.can_remove_block(*block_id){
                    continue;
                };
                let block = if (*field_id==1) {self.field1.blocks.get(block_id).unwrap()} else {self.field2.blocks.get(block_id).unwrap()};
                // (xs, xe, ys, ye, zs, ze)
                let (xs, xe, ys, ye, zs, ze) = block.get_bbox();
                let cubes = block.get_cubes();
                for x in 0..(self.D+xs-xe){
                    for y in 0..(self.D+ys-ye){
                        for z in 0..(self.D+zs-ze){
                            if x==xs && y==ys && z==zs{
                                continue;
                            }
                            let mut is_ok = true;
                            for (x_cube, y_cube, z_cube) in cubes.iter(){
                                let (x_new_cube, y_new_cube, z_new_cube) = ((x_cube+x-xs), (y_cube+y-ys), (z_cube+z-zs));
                                let filled_block_id = field.filled_block_id[[x_new_cube, y_new_cube, z_new_cube]] as usize;
                                if !(filled_block_id==0 || filled_block_id==*block_id) || field.can_place[[x_new_cube, y_new_cube, z_new_cube]]==0_u16{
                                    is_ok = false;
                                    break;
                                }
                            }
                            if is_ok{
                                // let mut change_block_info = vec![];
                                let mut change_block_info_ = HashMap::new();
                                let new_block_id = field.get_new_block_id();
                                // eprintln!("{} {} {} {} s1:{} s2:{}", block_id_1, self.field1.filled_block_id[*xyz1_], block_id_2, self.field2.filled_block_id[*xyz2_], serial, serial_);
                                for (x_cube, y_cube, z_cube) in cubes.iter(){
                                    change_block_info_.insert((*x_cube, *y_cube, *z_cube), 0);
                                }
                                for (x_cube, y_cube, z_cube) in cubes.iter(){
                                    let (x_new_cube, y_new_cube, z_new_cube) = ((x_cube+x-xs), (y_cube+y-ys), (z_cube+z-zs));
                                    change_block_info_.insert((x_new_cube, y_new_cube, z_new_cube), new_block_id);
                                }
                                let mut change_block_info = vec![];
                                for (k, v) in change_block_info_.iter(){
                                    change_block_info.push((*v, *k));
                                }
                                if *field_id==1{
                                    return (change_block_info, vec![]);
                                } else {
                                    return (vec![], change_block_info);
                                }
                            }
                        }
                    }
                }
            }
            (vec![], vec![])
        }
    }

    fn get_neighbor2(&mut self) -> (Vec<(usize, (usize, usize, usize))>, Vec<(usize, (usize, usize, usize))>){
        let mut rng = rand::thread_rng();
        for _ in 0..100000{
            let xyz1 = (rng.gen::<usize>()%self.D, rng.gen::<usize>()%self.D, rng.gen::<usize>()%self.D);
            let xyz2 = (rng.gen::<usize>()%self.D, rng.gen::<usize>()%self.D, rng.gen::<usize>()%self.D);
            let dir1 = rng.gen::<usize>()%6;
            let dir2 = rng.gen::<usize>()%6;
            let xyz1_opt = get_place_from_dir(xyz1, dir1, self.D);
            let xyz2_opt = get_place_from_dir(xyz2, dir2, self.D);
            if let (Some(xyz1_), Some(xyz2_)) = (xyz1_opt, xyz2_opt){
                if self.field1.can_place[xyz1]==0 || self.field1.can_place[xyz1_]==0 || self.field2.can_place[xyz2]==0 || self.field2.can_place[xyz2_]==0{
                    continue;
                }
                let serials1 = self.field1.get_serials(&vec![xyz1, xyz1_]);
                let serials2 = self.field2.get_serials(&vec![xyz2, xyz2_]);
                if serials1==serials2{
                    let (serial1, serial1_) = (self.field1.get_serial(xyz1), self.field1.get_serial(xyz1_));
                    let (serial2, serial2_) = (self.field2.get_serial(xyz2), self.field2.get_serial(xyz2_));
                    if serial1==&self.field1.none_block_serial && serial1_==&self.field1.none_block_serial{
                        continue;
                    } else if serial1==&self.field1.none_block_serial {
                        if self.field1.get_serial(xyz1)==self.field2.get_serial(xyz2){
                            return (vec![(self.field1.filled_block_id[xyz1_] as usize, xyz1)],
                                    vec![(self.field2.filled_block_id[xyz2_] as usize, xyz2)])
                        } else {
                            return (vec![(self.field1.filled_block_id[xyz1_] as usize, xyz1)],
                                    vec![(self.field2.filled_block_id[xyz2] as usize, xyz2_)])
                        }
                    } else if self.field1.get_serial(xyz1)==self.field2.get_serial(xyz2){
                        return (vec![(self.field1.filled_block_id[xyz1] as usize, xyz1_)],
                                vec![(self.field2.filled_block_id[xyz2] as usize, xyz2_)])
                    } else {
                        return (vec![(self.field1.filled_block_id[xyz1] as usize, xyz1_)],
                                vec![(self.field2.filled_block_id[xyz2_] as usize, xyz2)])
                    }
                }
            }
        }
        (vec![], vec![])
    }

    fn get_raw_score(&mut self) -> i64{
        let mut score: i64 = 0;
        for block in self.field1.blocks.values(){
            if block.n_cube>0{
                score += 1000000000/block.n_cube as i64;
            }
        }
        score
    }

    fn get_score(&mut self, mode: usize) -> i64{
        if mode==0{
            self.get_my_score()
        } else {
            self.get_raw_score()
        }
    }

    fn get_my_score(&mut self) -> i64{
        let mut score: i64 = (self.field1.blocks.len()*100000) as i64;
        for block in self.field1.blocks.values(){
            if block.n_cube>0{
                score -= (block.n_cube as i64).pow(4);
            }
        }
        score
    }


    fn print(&self){
        // idの振り直し
        let n_blocks = self.field1.blocks.len();
        let mut oldid2newid_1: HashMap<usize, usize> = HashMap::new();
        let mut oldid2newid_2: HashMap<usize, usize> = HashMap::new();
        let mut unused_1: HashSet<usize> = HashSet::new();
        oldid2newid_1.insert(0, 0);
        oldid2newid_2.insert(0, 0);
        for (new_id, old_id) in self.field1.blocks.keys().enumerate(){
            oldid2newid_1.insert(*old_id, new_id+1);
            unused_1.insert(*old_id);
        }
        for (old_id, block) in self.field2.blocks.iter(){
            for (old_id1, new_id1) in oldid2newid_1.iter(){
                if unused_1.contains(old_id1) && self.field1.blocks.get(old_id1).unwrap().serial == block.serial {
                    oldid2newid_2.insert(*old_id, *new_id1);
                    unused_1.remove(old_id1);
                    break;
                }
            }
        }
        // eprintln!("{:?} {:?}", self.field1.filled_block_id, self.field2.filled_block_id);
        let mut v1: Vec<&String> = self.field1.blocks.iter().map(|x| &x.1.serial).collect();
        let mut v2: Vec<&String> = self.field2.blocks.iter().map(|x| &x.1.serial).collect();
        v1.sort();
        v2.sort();
        // eprintln!("{:?} {:?}", v1, v2);
        // eprintln!("{:?}", v1==v2);
        // eprintln!("{:?} {:?}", oldid2newid_1, oldid2newid_2);
        // eprintln!("{:?}", oldid2newid_1==oldid2newid_2);
        // eprintln!("{:?} {:?}", oldid2newid_1.len(), oldid2newid_2.len());
        // eprintln!("{:?} {:?}", self.field1.blocks.len(), self.field2.blocks.len());
        println!("{}", n_blocks);
        let dst: Vec<String> = self.field1.filled_block_id.iter().map(|x| oldid2newid_1.get(&(*x as usize)).unwrap().to_string()).collect();
        println!("{}", dst.join(" "));
        let dst: Vec<String> = self.field2.filled_block_id.iter().map(|x| oldid2newid_2.get(&(*x as usize)).unwrap().to_string()).collect();
        println!("{}", dst.join(" "));
    }

}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut movecnt = 0;
    let mut swapcnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 1000000.0;
    let end_temp: f64 = 0.1;
    let mut temp = start_temp;
    eprintln!("simanneal start at {}", timer.elapsed().as_secs_f64());
    loop {
        let elasped_time = timer.elapsed().as_secs_f64();
        if elasped_time >= limit{
            break;
        }
        all_iter += 1;
        // let mode: usize = 0;
        let mode: usize = if (elasped_time <= limit*3.0/4.0) {0} else {1};
        let pre_score = state.get_score(mode);
        let mut neighbor = state.get_neighbor();
        let is_success = state.update_if_ok(&neighbor.0, &neighbor.1);
        if !is_success{
            continue;
        }
        let new_score = state.get_score(mode);
        let score_diff = new_score-pre_score;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if (score_diff<=0 || ((-score_diff as f64)/temp).exp() > rng.gen::<f64>()){
            accepted_cnt += 1;
            // eprintln!("{}", new_score);
            if state.get_score(mode)<best_state.get_score(mode){
                best_state = state.clone();
            }
        } else {
            state.undo();
            // state.undo(input, params);
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("");
    best_state
}


fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 5.5);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut state = State::init_state(input);
    let best_state = simanneal(input, state, timer, tl);
    best_state.print();
    // eprintln!("{:?}", state);
    // let mut block1 = Block::single_cube(0, (1, 1, 1));
    // block1.update_cubes(&vec![(1, (1, 2, 1)), (1, (1, 2, 2)), (0, (1, 1, 1))]);
    // //block1.add_cubes(&vec![(1, 2, 1), (2, 1, 1), (1, 1, 2)]);
    // let mut block2 = Block::single_cube(0, (8, 8, 8));
    // block2.add_cubes(&vec![(8, 7, 8), (8, 8, 7), (7, 8, 8)]);
    // eprintln!("{:?} {}", block1, block1.serial);
    // eprintln!("{:?} {}", block2, block2.serial);
}

// fn solve(input: &Input, timer:&Instant, tl: f64) -> State{
//     let init_state = State::init_state(input);
//     let best_state = simanneal(input, init_state, &timer, tl);
//     //println!("{}", best_state);
//     eprintln!("{}", timer.elapsed().as_secs_f64());
//     best_state
// }
