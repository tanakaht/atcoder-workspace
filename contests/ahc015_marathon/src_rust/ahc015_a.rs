use proconio::*;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::cmp;

const DIR: [char; 4] = ['F', 'R', 'B', 'L'];

#[derive(Debug, Clone)]
struct State {
    P: [u8; 100],
    F: [u8; 100],
    cnt: usize
}

impl State{
    fn new(P: [u8; 100], F: [u8; 100] , cnt: usize) -> Self{
        Self {
            P,
            F,
            cnt
        }
    }

    fn add(&mut self, i: usize, f: u8){
        self.P[i] = f;
    }

    fn moves(&mut self, dir: char){
        if dir=='F'{
            for j in 0..10{
                let mut fs_j: Vec<u8> = Vec::new();
                for i in 0..10{
                    if self.P[i*10+j]!=0{
                        fs_j.push(self.P[i*10+j])
                    }
                }
                for _ in 0..10-fs_j.len(){
                    fs_j.push(0)
                }
                for (i, f) in fs_j.iter().enumerate(){
                    self.P[i*10+j] = *f;
                }
            }
        } else if dir=='B'{
            for j in 0..10{
                let mut fs_j: Vec<u8> = Vec::new();
                for i in 0..10{
                    if self.P[i*10+j]!=0{
                        fs_j.push(self.P[i*10+j])
                    }
                }
                fs_j.reverse();
                for _ in 0..10-fs_j.len(){
                    fs_j.push(0)
                }
                fs_j.reverse();
                for (i, f) in fs_j.iter().enumerate(){
                    self.P[i*10+j] = *f;
                }
            }
        } else if dir=='L'{
            for i in 0..10{
                let mut fs_i: Vec<u8> = Vec::new();
                for j in 0..10{
                    if self.P[i*10+j]!=0{
                        fs_i.push(self.P[i*10+j])
                    }
                }
                for _ in 0..10-fs_i.len(){
                    fs_i.push(0)
                }
                for (j, f) in fs_i.iter().enumerate(){
                    self.P[i*10+j] = *f;
                }
            }
        } else if dir=='R'{
            for i in 0..10{
                let mut fs_i: Vec<u8> = Vec::new();
                for j in 0..10{
                    if self.P[i*10+j]!=0{
                        fs_i.push(self.P[i*10+j])
                    }
                }
                fs_i.reverse();
                for _ in 0..10-fs_i.len(){
                    fs_i.push(0)
                }
                fs_i.reverse();
                for (j, f) in fs_i.iter().enumerate(){
                    self.P[i*10+j] = *f;
                }
            }
        }
    }

    fn get_neighbor_dir(&mut self) -> Vec<(char, State)>{
        let mut ret: Vec<(char, State)> = Vec::new();
        for dir in DIR.iter(){
            let mut state = self.clone();
            state.moves(*dir);
            ret.push((*dir, state));
        }
        return ret;
    }

    fn get_score(&mut self) -> f32{
        return 0.0;
    }
}

#[derive(Debug, Clone)]
struct StateNode{
    state: State,
    is_moveturn: bool,
    children: Vec<StateNode>,
}

impl StateNode{
    fn new(state: State, is_moveturn: bool, children: Vec<StateNode>) -> Self{
        StateNode { state, is_moveturn, children}
    }

    fn add_child(&mut self, child: &StateNode){
        self.children.push(child.clone());
    }

    fn score(&mut self) -> f32{
        if self.children.len()==0{
            return self.state.get_score();
        } else if self.is_moveturn {
            let mut score: f32 = 0.0;
            for child in self.children.iter_mut(){
                let child_score = child.score();
                if score < child_score{
                    score = child_score
                }
            }
            return score;
        } else {
            let mut score: f32 = 0.0;
            for child in self.children.iter_mut(){
                let child_score = child.score();
                score += child_score;
            }
            return score;
        }
    }
}

struct StateTree{
    root_node: StateNode,
    child_nodes: Vec<(char, StateNode)>
}

impl StateTree{
    fn new(root_state: &mut State, depth: usize, n_neighbor: usize) -> Self{
        let root_node = StateNode::new(root_state.clone(), true, vec![]);
        let mut child_nodes = vec![];
        for (dir, state) in root_state.get_neighbor_dir(){
            let node =  StateNode::new(state, false, vec![]);
            child_nodes.push((dir, node))
        }
        // tree構築

        Self{
            root_node,
            child_nodes
        }
    }
    fn get_best_move(&mut self) -> char{
        let mut best_score:f32 = 0.0;
        let mut best_dir: char = 'F';
        for (dir, state) in self.child_nodes.iter_mut(){
            let state_score = state.score();
            if state_score > best_score{
                best_score = state_score;
                best_dir = *dir;
            }
        }
        return best_dir;
    }
}


fn main() {
    input! {
        F_: [u8; 100]
    }
    const F: [u8; 100] = [0; 100];
    for (i, v) in F_.iter().enumerate(){
        F[i] = *v;
    }
    let P = [0; 100];
    let mut state = State::new(P, F, 0);
    for turn in 0..100{
        input!{
            i: usize
        }
        state.add(i, F[i]);
        let mut tree = StateTree::new(&mut state, 2, 5);
        let dir = tree.get_best_move();
        state.moves(dir);
        println!("{}", dir);
    }
}
