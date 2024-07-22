use proconio::input;
use rand::{Rng, SeedableRng};
use rand_pcg::{Mcg128Xsl64, Pcg64Mcg};
use std::{collections::HashMap, process::exit};

pub trait ChangeMinMax {
    fn chmin(&mut self, x: Self) -> bool;
    fn chmax(&mut self, x: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn chmin(&mut self, x: T) -> bool {
        *self > x && {
            *self = x;
            true
        }
    }

    fn chmax(&mut self, x: T) -> bool {
        *self < x && {
            *self = x;
            true
        }
    }
}

// Problem Parameters
const TIME_LIMIT_SEC: f64 = 2.9;
const GRID_SIZE: usize = 5;
const NUM_CONTAINERS: usize = GRID_SIZE * GRID_SIZE;
const ACTION_CHARS: [char; 8] = ['U', 'D', 'L', 'R', '.', 'B', 'P', 'Q'];

// Auxiliary Parameters
const RECEIVE_PATTERNS: usize = (GRID_SIZE + 1).pow(GRID_SIZE as u32);
const U_MASK: u32 = 0b_11111_11111_11111_11111_00000;
const D_MASK: u32 = 0b_00000_11111_11111_11111_11111;
const L_MASK: u32 = 0b_11110_11110_11110_11110_11110;
const R_MASK: u32 = 0b_01111_01111_01111_01111_01111;
const UDLR_MASKS: [u32; 4] = [U_MASK, D_MASK, L_MASK, R_MASK];

// Hyper Parameters
const INTERNAL_MAX_TURN: usize = 100;
const BFS_ABORT_TURN: usize = 20;
const STACK_POS_COSTS: [usize; GRID_SIZE + 1] = [0, 1, 3, 6, 10, 15];
const LINEAR_CONFLICT_COST: usize = 4;
const NUM_STACKS_COST: usize = 2;
const BIG_CRANE_COST: usize = 5;
const STACK_Y: [usize; 2] = [2, 3];
const BEAM_WIDTHS: [usize; 5] = [27000, 25000, 22000, 18000, 10000];
const WIDTH_TIME_MSEC: [usize; 5] = [1500, 2000, 2500, 2700, usize::MAX];
const BEAM_WIDTH: usize = BEAM_WIDTHS[0];
const DIRECTIONS_CAPACITY: usize = 1 << 21;

type Input = [[usize; GRID_SIZE]; GRID_SIZE];
type Output = [Vec<Action>; GRID_SIZE];

fn main() {
    get_time_sec();

    let input = read_input();
    let env = Environment::new(input);

    let mut curr_states = Vec::with_capacity(BEAM_WIDTH);
    let mut next_states = Vec::with_capacity(BEAM_WIDTH);
    let mut best_score = usize::MAX;
    const EMPTY_ACTION_VEC: Vec<Action> = Vec::new();
    let mut best_output = [EMPTY_ACTION_VEC; GRID_SIZE];
    let mut selector = Selector::new(BEAM_WIDTH);
    let mut memory_pool = MemoryPool::new();

    curr_states.push(State::new(&env));
    while best_score == usize::MAX || get_time_sec() < TIME_LIMIT_SEC {
        selector.clear();
        selector.n =
            BEAM_WIDTHS[match WIDTH_TIME_MSEC.binary_search(&((1e3 * get_time_sec()) as usize)) {
                Ok(index) => index,
                Err(index) => index,
            }];
        memory_pool.directions.clear();
        for state_id in 0..curr_states.len() {
            let state = &mut curr_states[state_id];
            if state.is_completed() {
                if best_score.chmin(state.max_actions_len) {
                    best_output = state.actions.clone();
                }
            } else {
                state.push_candidates(state_id, best_score, &mut selector, &env, &mut memory_pool);
            }
        }
        let candidates = selector.select();
        for i in 0..candidates.len() {
            let candidate = &candidates[i];
            if i < next_states.len() {
                curr_states[candidate.state_id as usize].copy_to(&mut next_states[i]);
            } else {
                next_states.push(curr_states[candidate.state_id as usize].clone());
            }
            next_states[i].act(candidate, &env, &memory_pool);
        }
        if next_states.len() > candidates.len() {
            next_states.truncate(candidates.len());
        }
        std::mem::swap(&mut curr_states, &mut next_states);

        if curr_states.is_empty() {
            if best_score == usize::MAX {
                print_output(&next_states[0].actions);
                panic!("all states are dead");
            }
            break;
        }
    }

    print_output(&best_output);

    exit(0);
}

fn read_input() -> Input {
    input! {
        n: usize,
        a: [[usize; GRID_SIZE]; GRID_SIZE],
    };
    debug_assert_eq!(n, GRID_SIZE);
    let mut ret = [[0; GRID_SIZE]; GRID_SIZE];
    for x in 0..GRID_SIZE {
        ret[x] = a[x].clone().try_into().unwrap();
    }
    ret
}

fn inverse_input(input: &Input) -> [(usize, usize); NUM_CONTAINERS] {
    let mut ret = [(0, 0); NUM_CONTAINERS];
    for x in 0..GRID_SIZE {
        for i in 0..GRID_SIZE {
            ret[input[x][i]] = (x, i);
        }
    }
    ret
}

fn print_output(output: &Output) {
    let mut score = 0;
    for crane in 0..GRID_SIZE {
        score.chmax(output[crane].len());
    }
    for crane in 0..GRID_SIZE {
        for action in output[crane].iter() {
            print!("{}", ACTION_CHARS[*action as usize]);
        }
        if output[crane].len() < score {
            print!("{}", ACTION_CHARS[Action::Bomb as usize]);
        }
        print!("\n");
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Action {
    Up,
    Down,
    Left,
    Right,
    Stay,
    Bomb,
    PickUp,
    Release,
}

struct Environment {
    receive_order: Input,
    receive_pos: [(usize, usize); NUM_CONTAINERS],
    receive_cost: [usize; RECEIVE_PATTERNS],
    stack_hashes: [[u64; NUM_CONTAINERS]; NUM_CONTAINERS],
    crane_hashes: [u64; NUM_CONTAINERS],
    receive_hashes: [u64; NUM_CONTAINERS],
}

impl Environment {
    fn new(input: Input) -> Environment {
        let mut env = Environment {
            receive_order: input,
            receive_pos: inverse_input(&input),
            receive_cost: [0; RECEIVE_PATTERNS],
            stack_hashes: [[0; NUM_CONTAINERS]; NUM_CONTAINERS],
            crane_hashes: [0; NUM_CONTAINERS],
            receive_hashes: [0; NUM_CONTAINERS],
        };
        let mut rng = Pcg64Mcg::seed_from_u64(0);
        env.init_receiving_cost();
        env.init_stack_hashes(&mut rng);
        env.init_crane_hashes(&mut rng);
        env.init_receive_hashes(&mut rng);
        env
    }

    fn init_receiving_cost(&mut self) {
        for hash in (0..(RECEIVE_PATTERNS - 1)).rev() {
            let mut min_cost = usize::MAX;
            let mut receive_progress = Environment::decode_to_receive_progress(hash);
            let dispatch_progress = self.calc_dispatch_progress(&receive_progress);
            for x in 0..GRID_SIZE {
                if receive_progress[x] as usize == GRID_SIZE {
                    continue;
                }
                let container = self.receive_order[x][receive_progress[x] as usize];
                receive_progress[x] += 1;
                let mut cost = self.get_cost(&receive_progress)
                    + 2 * GRID_SIZE
                    + x.abs_diff(container / GRID_SIZE);
                receive_progress[x] -= 1;
                if container > dispatch_progress[container / GRID_SIZE] as usize {
                    // have to stack the container
                    cost += 2;
                }
                min_cost.chmin(cost);
            }
            self.receive_cost[hash] = min_cost;
        }
    }

    fn get_cost(&self, receive_progress: &[u8; GRID_SIZE]) -> usize {
        self.receive_cost[Environment::encode_receive_progress(receive_progress)]
    }

    fn calc_dispatch_progress(&self, receive_progress: &[u8; GRID_SIZE]) -> [u8; GRID_SIZE] {
        let mut dispatch_progress = [0; GRID_SIZE];
        for x in 0..GRID_SIZE {
            let mut container = GRID_SIZE * x;
            while container < GRID_SIZE * (x + 1) {
                let (receive_x, receive_i) = self.receive_pos[container];
                if receive_progress[receive_x] as usize <= receive_i {
                    break;
                }
                container += 1;
            }
            dispatch_progress[x] = container as u8;
        }
        dispatch_progress
    }

    fn encode_receive_progress(receive_progress: &[u8; GRID_SIZE]) -> usize {
        let mut ret = 0;
        for &progress in receive_progress.iter() {
            debug_assert!(progress as usize <= GRID_SIZE);
            ret *= GRID_SIZE + 1;
            ret += progress as usize;
        }
        ret
    }

    fn decode_to_receive_progress(mut hash: usize) -> [u8; GRID_SIZE] {
        let mut ret = [0; GRID_SIZE];
        for x in (0..GRID_SIZE).rev() {
            ret[x] = (hash % (GRID_SIZE + 1)) as u8;
            hash /= GRID_SIZE + 1;
        }
        ret
    }

    fn init_stack_hashes(&mut self, rng: &mut Mcg128Xsl64) {
        for pos in 0..NUM_CONTAINERS {
            for container in 0..NUM_CONTAINERS {
                self.stack_hashes[pos][container] = rng.gen();
            }
        }
    }

    fn get_stack_hash(&self, pos: usize, container: usize) -> u64 {
        self.stack_hashes[pos][container]
    }

    fn init_crane_hashes(&mut self, rng: &mut Mcg128Xsl64) {
        for pos in 0..NUM_CONTAINERS {
            self.crane_hashes[pos] = rng.gen();
        }
    }

    fn get_crane_hash(&self, pos: usize, turn: usize) -> u64 {
        self.crane_hashes[pos].wrapping_mul(turn as u64)
    }

    fn init_receive_hashes(&mut self, rng: &mut Mcg128Xsl64) {
        for container in 0..NUM_CONTAINERS {
            self.receive_hashes[container] = rng.gen();
        }
    }

    fn get_receive_hash(&self, container: usize) -> u64 {
        self.receive_hashes[container]
    }
}

struct MemoryPool {
    directions: Vec<Action>,
    pick_up_actions: Vec<Action>,
    pick_up_visited: [u32; INTERNAL_MAX_TURN + 1],
    release_visited: [u32; INTERNAL_MAX_TURN + 1],
}

impl MemoryPool {
    fn new() -> MemoryPool {
        MemoryPool {
            directions: Vec::with_capacity(DIRECTIONS_CAPACITY),
            pick_up_actions: Vec::new(),
            pick_up_visited: [0; INTERNAL_MAX_TURN + 1],
            release_visited: [0; INTERNAL_MAX_TURN + 1],
        }
    }
}

#[derive(Clone)]
struct State {
    max_actions_len: usize,
    sum_actions_len: usize,
    stack_cost: usize,
    crane_pos_bonus: usize,
    actions: Output,
    crane_pos: [u8; GRID_SIZE],
    grid: [u8; NUM_CONTAINERS],
    receive_progress: [u8; GRID_SIZE],
    dispatch_progress: [u8; GRID_SIZE],
    used_by_crane: [u32; INTERNAL_MAX_TURN + 1],
    used_by_container: Vec<u32>,
    masks: [[u32; 4]; INTERNAL_MAX_TURN + 1],
    stack_pos: u32,
    zobrist_hash: u64,
    receive_turn: [u8; GRID_SIZE],
    dispatch_turn: [u8; GRID_SIZE],
    stack_turn: [u8; NUM_CONTAINERS],
    last_turn: [u8; NUM_CONTAINERS],
}

impl State {
    fn new(env: &Environment) -> State {
        let mut crane_pos = [0; GRID_SIZE];
        for x in 0..GRID_SIZE {
            crane_pos[x] = (GRID_SIZE * x) as u8;
        }
        let mut grid = [NUM_CONTAINERS as u8; NUM_CONTAINERS];
        for x in 0..GRID_SIZE {
            grid[GRID_SIZE * x] = env.receive_order[x][0] as u8;
        }
        let mut dispatch_progress = [0; GRID_SIZE];
        for x in 0..GRID_SIZE {
            dispatch_progress[x] = (GRID_SIZE * x) as u8;
        }
        let mut stack_pos = 0;
        for x in 0..GRID_SIZE {
            bit_set(&mut stack_pos, GRID_SIZE * x);
        }
        const EMPTY_ACTION_VEC: Vec<Action> = Vec::new();
        State {
            max_actions_len: 0,
            sum_actions_len: 0,
            stack_cost: 0,
            crane_pos_bonus: (GRID_SIZE - 1) * GRID_SIZE,
            actions: [EMPTY_ACTION_VEC; GRID_SIZE],
            crane_pos,
            grid,
            receive_progress: [0; GRID_SIZE],
            dispatch_progress,
            used_by_crane: [0; INTERNAL_MAX_TURN + 1],
            used_by_container: Vec::with_capacity(INTERNAL_MAX_TURN),
            masks: [UDLR_MASKS; INTERNAL_MAX_TURN + 1],
            stack_pos,
            zobrist_hash: 0,
            receive_turn: [0; GRID_SIZE],
            dispatch_turn: [0; GRID_SIZE],
            stack_turn: [0; NUM_CONTAINERS],
            last_turn: [0; NUM_CONTAINERS],
        }
    }

    fn copy_to(&self, state: &mut State) {
        if state.max_actions_len > self.max_actions_len {
            state.used_by_crane[self.max_actions_len..=state.max_actions_len].fill(0);
            state.masks[self.max_actions_len..=state.max_actions_len].fill(UDLR_MASKS);
        }
        let mut min_actions_len = usize::MAX;
        for crane in 0..GRID_SIZE {
            min_actions_len.chmin(self.actions[crane].len());
        }
        if min_actions_len > 0 {
            min_actions_len -= 1;
        }
        state.max_actions_len = self.max_actions_len;
        state.sum_actions_len = self.sum_actions_len;
        state.stack_cost = self.stack_cost;
        state.crane_pos_bonus = self.crane_pos_bonus;
        for crane in 0..GRID_SIZE {
            state.actions[crane].resize(self.actions[crane].len(), Action::Up);
            state.actions[crane].copy_from_slice(&self.actions[crane]);
        }
        state.crane_pos = self.crane_pos;
        state.grid = self.grid;
        state.receive_progress = self.receive_progress;
        state.dispatch_progress = self.dispatch_progress;
        state.used_by_crane[min_actions_len..=self.max_actions_len]
            .copy_from_slice(&self.used_by_crane[min_actions_len..=self.max_actions_len]);
        state.masks[min_actions_len..=self.max_actions_len]
            .copy_from_slice(&self.masks[min_actions_len..=self.max_actions_len]);
        state
            .used_by_container
            .resize(self.used_by_container.len(), 0);
        state.used_by_container[min_actions_len..]
            .copy_from_slice(&self.used_by_container[min_actions_len..]);
        state.stack_pos = self.stack_pos;
        state.zobrist_hash = self.zobrist_hash;
        state.receive_turn = self.receive_turn;
        state.dispatch_turn = self.dispatch_turn;
        state.stack_turn = self.stack_turn;
        state.last_turn = self.last_turn;
    }

    fn is_completed(&self) -> bool {
        for x in 0..GRID_SIZE {
            if (self.dispatch_progress[x] as usize) < GRID_SIZE * (x + 1) {
                return false;
            }
        }
        true
    }

    // the lower, the better
    // for debug
    fn evaluate(&self, env: &Environment) -> u16 {
        (GRID_SIZE * self.max_actions_len).max(
            self.sum_actions_len
                + self.actions[0].len() / BIG_CRANE_COST
                + env.get_cost(&self.receive_progress)
                + self.stack_cost
                + self.get_stack_pos_cost()
                - self.crane_pos_bonus,
        ) as u16
    }

    fn get_stack_pos_cost(&self) -> usize {
        let mut num_stack_rows = 0;
        let mut linear_conflicts = 0;
        let mut num_stacks = 0;
        for x in 0..GRID_SIZE {
            for y in STACK_Y {
                let pos = GRID_SIZE * x + y;
                if (self.grid[pos] as usize) < NUM_CONTAINERS {
                    num_stack_rows += 1;
                    num_stacks += 1;
                    if (self.grid[pos + 1] as usize) < NUM_CONTAINERS {
                        num_stacks += 1;
                        let container_l = self.grid[pos] as usize;
                        let container_r = self.grid[pos + 1] as usize;
                        if container_l / GRID_SIZE == container_r / GRID_SIZE
                            && container_l % GRID_SIZE < container_r % GRID_SIZE
                        {
                            linear_conflicts += 1;
                        }
                    }
                    break;
                }
            }
        }
        STACK_POS_COSTS[num_stack_rows]
            + LINEAR_CONFLICT_COST * linear_conflicts
            + num_stacks / NUM_STACKS_COST
    }

    fn evaluate_next_state(
        &mut self,
        crane: usize,
        num_actions: usize,
        pick_up_pos: usize,
        release_pos: usize,
        container: usize,
        env: &Environment,
    ) -> u16 {
        let new_sum_actions_len = self.sum_actions_len + num_actions;
        let new_big_crane_cost = if crane == 0 {
            (self.actions[0].len() + num_actions) / BIG_CRANE_COST
        } else {
            self.actions[0].len() / BIG_CRANE_COST
        };
        let receive = pick_up_pos % GRID_SIZE == 0;
        let dispatch = release_pos % GRID_SIZE == GRID_SIZE - 1;
        let new_receive_cost = if receive {
            let (receive_x, receive_i) = env.receive_pos[container];
            debug_assert_eq!(self.receive_progress[receive_x] as usize, receive_i);
            self.receive_progress[receive_x] += 1;
            let receive_cost = env.get_cost(&self.receive_progress);
            self.receive_progress[receive_x] -= 1;
            receive_cost
        } else {
            env.get_cost(&self.receive_progress)
        };
        let new_stack_cost = self.calc_new_stack_cost(container, pick_up_pos, release_pos);

        if receive && (!dispatch) {
            debug_assert_eq!(self.grid[release_pos] as usize, NUM_CONTAINERS);
            self.grid[release_pos] = container as u8;
        }
        if (!receive) && dispatch {
            debug_assert_eq!(self.grid[pick_up_pos] as usize, container);
            self.grid[pick_up_pos] = NUM_CONTAINERS as u8;
        }
        let new_stack_pos_cost = self.get_stack_pos_cost();
        if receive && (!dispatch) {
            self.grid[release_pos] = NUM_CONTAINERS as u8;
        }
        if (!receive) && dispatch {
            self.grid[pick_up_pos] = container as u8;
        }
        let new_crane_pos_bonus = self.crane_pos_bonus + self.crane_pos[crane] as usize % GRID_SIZE
            - release_pos % GRID_SIZE;
        let max_action_cost = GRID_SIZE
            * self
                .max_actions_len
                .max(self.actions[crane].len() + num_actions);
        max_action_cost.max(
            new_sum_actions_len
                + new_big_crane_cost
                + new_receive_cost
                + new_stack_cost
                + new_stack_pos_cost
                - new_crane_pos_bonus,
        ) as u16
    }

    fn calc_new_stack_cost(
        &self,
        container: usize,
        pick_up_pos: usize,
        release_pos: usize,
    ) -> usize {
        let receive = pick_up_pos % GRID_SIZE == 0;
        let dispatch = release_pos % GRID_SIZE == GRID_SIZE - 1;
        if receive {
            if dispatch {
                self.stack_cost
            } else {
                self.stack_cost
                    + manhattan(
                        release_pos / GRID_SIZE,
                        release_pos % GRID_SIZE,
                        container / GRID_SIZE,
                        GRID_SIZE - 1,
                    )
                    + 2
            }
        } else {
            debug_assert!(dispatch);
            self.stack_cost
                - manhattan(
                    pick_up_pos / GRID_SIZE,
                    pick_up_pos % GRID_SIZE,
                    container / GRID_SIZE,
                    GRID_SIZE - 1,
                )
                + 2
        }
    }

    fn get_new_hash(
        &self,
        pick_up_pos: usize,
        release_pos: usize,
        container: usize,
        env: &Environment,
    ) -> u64 {
        let receive = pick_up_pos % GRID_SIZE == 0;
        let dispatch = release_pos % GRID_SIZE == GRID_SIZE - 1;
        let mut ret = self.zobrist_hash;
        if receive && (!dispatch) {
            ret ^= env.get_stack_hash(release_pos, container);
        }
        if (!receive) && dispatch {
            ret ^= env.get_stack_hash(pick_up_pos, container);
        }
        if dispatch {
            ret ^= env.get_receive_hash(container);
        }
        ret
    }

    fn get_crane_hash(
        &self,
        crane: usize,
        num_actions: usize,
        release_pos: usize,
        env: &Environment,
    ) -> u64 {
        let mut ret = 0;
        for c in 0..GRID_SIZE {
            let x = if c == crane {
                env.get_crane_hash(release_pos, self.actions[crane].len() + num_actions)
            } else {
                env.get_crane_hash(self.crane_pos[crane] as usize, self.actions[crane].len())
            };
            if c == 0 {
                ret ^= x >> 1;
            } else {
                ret ^= x;
            }
        }
        ret
    }

    fn push_candidates(
        &mut self,
        state_id: usize,
        best_score: usize,
        selector: &mut Selector,
        env: &Environment,
        memory_pool: &mut MemoryPool,
    ) {
        if self.max_actions_len >= best_score || self.has_dead_crane() {
            return;
        }
        let (bfs_start_turn, bfs_end_turn) =
            self.bfs_with_all_cranes(&mut memory_pool.pick_up_visited);

        for pick_up_pos in 0..NUM_CONTAINERS {
            if self.grid[pick_up_pos] as usize == NUM_CONTAINERS {
                continue;
            }
            let container = self.grid[pick_up_pos] as usize;
            let receive = pick_up_pos % GRID_SIZE == 0;
            let can_dispatch = self.dispatch_progress[container / GRID_SIZE] as usize == container;
            if !receive && !can_dispatch {
                // the container is stacked and cannot dispatch
                continue;
            }
            // pick up
            let lower_goal_turn = if receive {
                self.receive_turn[pick_up_pos / GRID_SIZE] as usize
            } else {
                self.stack_turn[pick_up_pos] as usize
            };
            let (crane, end_turn, pick_up_then_release_flag) = if let Some(crane) = (0..GRID_SIZE)
                .find(|&crane| {
                    self.crane_pos[crane] as usize == pick_up_pos
                        && !self.actions[crane].is_empty()
                        && self.actions[crane].len() >= lower_goal_turn
                        && *self.actions[crane].last().unwrap() == Action::Release
                }) {
                self.pick_up_then_release(crane, pick_up_pos);
                (crane, self.actions[crane].len() - 1, true)
            } else {
                if let Some((crane, end_turn)) = self.get_pick_up_crane(
                    bfs_start_turn,
                    bfs_end_turn,
                    pick_up_pos,
                    lower_goal_turn,
                    &memory_pool.pick_up_visited,
                    &mut memory_pool.pick_up_actions,
                ) {
                    (crane, end_turn, false)
                } else {
                    continue;
                }
            };
            // release
            if can_dispatch {
                let release_pos = GRID_SIZE * (container / GRID_SIZE) + GRID_SIZE - 1;
                if self.release_then_push(
                    crane,
                    end_turn + 1,
                    pick_up_pos,
                    release_pos,
                    container,
                    best_score,
                    state_id,
                    selector,
                    env,
                    memory_pool,
                    pick_up_then_release_flag,
                ) {
                    if pick_up_then_release_flag {
                        self.revert_pick_up_then_release(crane, pick_up_pos);
                    }
                    continue;
                }
            }
            if !receive {
                if pick_up_then_release_flag {
                    self.revert_pick_up_then_release(crane, pick_up_pos);
                }
                continue;
            }
            for release_x in 0..GRID_SIZE {
                for release_y in STACK_Y {
                    // stack the container
                    let release_pos = GRID_SIZE * release_x + release_y;
                    if !self.can_stack(release_pos) {
                        continue;
                    }
                    self.release_then_push(
                        crane,
                        end_turn + 1,
                        pick_up_pos,
                        release_pos,
                        container,
                        best_score,
                        state_id,
                        selector,
                        env,
                        memory_pool,
                        pick_up_then_release_flag,
                    );
                }
            }
            if pick_up_then_release_flag {
                self.revert_pick_up_then_release(crane, pick_up_pos);
            }
        }
    }

    fn has_dead_crane(&mut self) -> bool {
        for crane in 0..GRID_SIZE {
            if self.calc_next_reachable(1 << self.crane_pos[crane], self.actions[crane].len(), true)
                == 0
            {
                return true;
            }
        }
        false
    }

    fn pick_up_then_release(&mut self, crane: usize, pick_up_pos: usize) {
        let pos_bit = 1 << pick_up_pos;

        let turn = self.actions[crane].len();
        self.pop_stack(turn, pos_bit);
        debug_assert_ne!(self.used_by_crane[turn] & pos_bit, 0);
        self.used_by_crane[turn] &= !pos_bit;
        self.actions[crane].pop();
    }

    fn revert_pick_up_then_release(&mut self, crane: usize, pick_up_pos: usize) {
        let pos_bit = 1 << pick_up_pos;

        self.actions[crane].push(Action::Release);
        let turn = self.actions[crane].len();
        debug_assert_eq!(self.used_by_crane[turn] & pos_bit, 0);
        self.used_by_crane[turn] |= pos_bit;
        self.push_stack(turn, pos_bit);
    }

    fn release_then_push(
        &mut self,
        crane: usize,
        start_turn: usize,
        pick_up_pos: usize,
        release_pos: usize,
        container: usize,
        best_score: usize,
        state_id: usize,
        selector: &mut Selector,
        env: &Environment,
        memory_pool: &mut MemoryPool,
        pick_up_then_release_flag: bool,
    ) -> bool {
        let lower_goal_turn = if release_pos % GRID_SIZE == GRID_SIZE - 1 {
            // dispatch
            self.dispatch_turn[release_pos / GRID_SIZE] as usize
        } else {
            // stack
            self.last_turn[release_pos] as usize
        };
        let end_turn = if let Some(end_turn) = self.bfs(
            start_turn,
            pick_up_pos,
            release_pos,
            crane == 0,
            lower_goal_turn,
            &mut memory_pool.release_visited,
        ) {
            end_turn
        } else {
            // not connected
            return false;
        };
        let num_actions = end_turn + 1 - self.actions[crane].len();
        if end_turn + 1 >= best_score {
            return false;
        }
        let cost =
            self.evaluate_next_state(crane, num_actions, pick_up_pos, release_pos, container, env);
        if !selector.accept_cost(cost) {
            return false;
        }
        let hash = self.get_new_hash(pick_up_pos, release_pos, container, env)
            ^ self.get_crane_hash(crane, num_actions, release_pos, &env);
        if !selector.accept_hash(cost, hash) {
            return false;
        }
        let directions_id = memory_pool.directions.len();
        if !pick_up_then_release_flag {
            memory_pool
                .directions
                .extend(memory_pool.pick_up_actions.clone());
            memory_pool.directions.push(Action::PickUp);
        }
        self.reconstruct_path(
            start_turn,
            end_turn,
            pick_up_pos,
            release_pos,
            &mut memory_pool.directions,
            &memory_pool.release_visited,
        );
        memory_pool.directions.push(Action::Release);
        selector.push(
            Candidate {
                cost,
                state_id: state_id as u16,
                crane: crane as u8,
                num_directions: (memory_pool.directions.len() - directions_id) as u8,
                directions_id: directions_id as u32,
                candidate_id: 0,
            },
            hash,
        );
        true
    }

    fn can_stack(&self, pos: usize) -> bool {
        if bit_get(self.stack_pos, pos) {
            // already stacked
            return false;
        }
        true
    }

    // return (start_turn, end_turn)
    fn bfs_with_all_cranes(
        &mut self,
        visited: &mut [u32; INTERNAL_MAX_TURN + 1],
    ) -> (usize, usize) {
        let start_turn = (0..GRID_SIZE)
            .map(|crane| self.actions[crane].len())
            .min()
            .unwrap();
        let end_turn = (start_turn + BFS_ABORT_TURN).min(INTERNAL_MAX_TURN);
        visited[start_turn..=end_turn].fill(0);
        for crane in 0..GRID_SIZE {
            if self.actions[crane].len() <= end_turn {
                bit_set(
                    &mut visited[self.actions[crane].len()],
                    self.crane_pos[crane] as usize,
                );
            }
        }
        for turn in start_turn..end_turn {
            visited[turn + 1] |= self.calc_next_reachable(visited[turn], turn, true);
        }
        (start_turn, end_turn)
    }

    // return (crane, arrival_turn)
    fn get_pick_up_crane(
        &self,
        bfs_start_turn: usize,
        bfs_end_turn: usize,
        to_pos: usize,
        lower_goal_turn: usize,
        visited: &[u32; INTERNAL_MAX_TURN + 1],
        bfs_actions: &mut Vec<Action>,
    ) -> Option<(usize, usize)> {
        let to_pos_bit = 1 << to_pos;
        for turn in bfs_start_turn.max(lower_goal_turn)..=bfs_end_turn {
            if (visited[turn] & to_pos_bit) > 0 && (self.used_by_crane[turn + 1] & to_pos_bit) == 0
            {
                // path reconstruction
                let mut pos_bit = 1 << to_pos;
                let end_turn = turn;
                let mut start_turn = turn;
                bfs_actions.clear();
                while start_turn > 0 && (self.used_by_crane[start_turn] & pos_bit) == 0 {
                    let direction = self.moved_by_udlr(pos_bit, start_turn - 1, visited);
                    if direction == Action::Bomb {
                        break;
                    }
                    start_turn -= 1;
                    bfs_actions.push(direction);
                    pos_bit = move_pos_bit_reverse(pos_bit, direction);
                }
                bfs_actions.reverse();
                let pos = pos_bit.trailing_zeros() as usize;
                let crane = (0..GRID_SIZE)
                    .find(|&crane| {
                        self.actions[crane].len() == start_turn
                            && self.crane_pos[crane] as usize == pos
                    })
                    .unwrap();
                return Some((crane, end_turn));
            }
        }
        None
    }

    // The crane is at fr_pos on the start_turn.
    fn bfs(
        &mut self,
        start_turn: usize,
        fr_pos: usize,
        to_pos: usize,
        can_traverse: bool,
        lower_goal_turn: usize,
        visited: &mut [u32; INTERNAL_MAX_TURN + 1],
    ) -> Option<usize> {
        if start_turn >= INTERNAL_MAX_TURN {
            return None;
        }
        // BFS
        let fr_pos_bit = 1 << fr_pos;
        let to_pos_bit = 1 << to_pos;
        visited[start_turn] = fr_pos_bit;

        let mut turn = start_turn;
        let bfs_abort_turn = INTERNAL_MAX_TURN.min(start_turn + BFS_ABORT_TURN);
        loop {
            if (visited[turn] & to_pos_bit) > 0
                && turn >= lower_goal_turn
                && (self.used_by_crane[turn + 1] & to_pos_bit) == 0
            {
                // arrived
                break;
            }
            visited[turn + 1] = self.calc_next_reachable(visited[turn], turn, can_traverse);
            turn += 1;
            if turn >= bfs_abort_turn {
                // too many turns
                // may be non-connected
                return None;
            }
            if visited[turn] == 0 {
                // crane is dead
                return None;
            }
        }
        Some(turn)
    }

    fn reconstruct_path(
        &mut self,
        start_turn: usize,
        end_turn: usize,
        fr_pos: usize,
        to_pos: usize,
        bfs_actions: &mut Vec<Action>,
        visited: &[u32; INTERNAL_MAX_TURN + 1],
    ) {
        // path reconstruction
        let mut pos_bit = 1 << to_pos;
        let original_len = bfs_actions.len();
        bfs_actions.resize(original_len + end_turn - start_turn, Action::Up);
        for turn in (start_turn..end_turn).rev() {
            let direction = self.moved_by_udlr(pos_bit, turn, visited);
            bfs_actions[original_len - start_turn + turn] = direction;
            pos_bit = move_pos_bit_reverse(pos_bit, direction);
            debug_assert_ne!(visited[turn] & pos_bit, 0);
        }
        debug_assert_eq!(pos_bit, 1 << fr_pos);
    }

    fn is_used_by_container(&mut self, turn: usize) -> u32 {
        if turn < self.used_by_container.len() {
            self.used_by_container[turn]
        } else {
            self.used_by_container.resize(turn + 1, self.stack_pos);
            self.stack_pos
        }
    }

    fn calc_next_reachable(&mut self, reachable: u32, turn: usize, can_traverse: bool) -> u32 {
        let mut ret = reachable;
        ret |= (reachable & self.masks[turn][0]) >> GRID_SIZE;
        ret |= (reachable & self.masks[turn][1]) << GRID_SIZE;
        ret |= (reachable & self.masks[turn][2]) >> 1;
        ret |= (reachable & self.masks[turn][3]) << 1;
        ret &= !(self.used_by_crane[turn + 1]);
        if !can_traverse {
            ret &= !self.is_used_by_container(turn + 1);
        }
        ret
    }

    // for path reconstruction
    fn moved_by_udlr(
        &self,
        pos_bit: u32,
        turn: usize,
        visited: &[u32; INTERNAL_MAX_TURN + 1],
    ) -> Action {
        let reachable = visited[turn];
        if (reachable & pos_bit) > 0 {
            Action::Stay
        } else if (reachable & self.masks[turn][0] & (pos_bit << GRID_SIZE)) > 0 {
            Action::Up
        } else if (reachable & self.masks[turn][1] & (pos_bit >> GRID_SIZE)) > 0 {
            Action::Down
        } else if (reachable & self.masks[turn][2] & (pos_bit << 1)) > 0 {
            Action::Left
        } else if (reachable & self.masks[turn][3] & (pos_bit >> 1)) > 0 {
            Action::Right
        } else {
            Action::Bomb
        }
    }

    fn act(&mut self, candidate: &Candidate, env: &Environment, memory_pool: &MemoryPool) {
        let crane = candidate.crane as usize;

        // move crane
        let mut pos_bit = 1 << self.crane_pos[crane];
        let mut turn = self.actions[crane].len();
        let mut container = NUM_CONTAINERS;
        let mut pick_up_pos = NUM_CONTAINERS;
        let mut release_pos = NUM_CONTAINERS;
        let directions = &memory_pool.directions[(candidate.directions_id as usize)
            ..((candidate.directions_id + candidate.num_directions as u32) as usize)];
        if !directions.contains(&Action::PickUp) {
            // abbreviate release then pick up
            debug_assert_eq!(*self.actions[crane].last().unwrap(), Action::Release);
            debug_assert_ne!(self.used_by_crane[turn] & pos_bit, 0);
            self.used_by_crane[turn] &= !pos_bit;
            self.actions[crane].pop();

            debug_assert_eq!(pick_up_pos, NUM_CONTAINERS);
            pick_up_pos = (pos_bit as u32).trailing_zeros() as usize;
            debug_assert_eq!(container, NUM_CONTAINERS);
            debug_assert_ne!(self.grid[pick_up_pos] as usize, NUM_CONTAINERS);
            container = self.grid[pick_up_pos] as usize;
            debug_assert_ne!(pick_up_pos % GRID_SIZE, 0);
            // pick up a stacked container
            self.pop_stack(turn, pos_bit);
            self.grid[pick_up_pos] = NUM_CONTAINERS as u8;
            turn -= 1;
        }
        for action in directions.iter() {
            turn += 1;
            match action {
                Action::PickUp => {
                    debug_assert_eq!(pick_up_pos, NUM_CONTAINERS);
                    pick_up_pos = (pos_bit as u32).trailing_zeros() as usize;
                    debug_assert_eq!(container, NUM_CONTAINERS);
                    debug_assert_ne!(self.grid[pick_up_pos] as usize, NUM_CONTAINERS);
                    container = self.grid[pick_up_pos] as usize;
                    if pick_up_pos % GRID_SIZE == 0 {
                        // receive
                        let x = pick_up_pos / GRID_SIZE;
                        debug_assert_eq!(
                            env.receive_order[x][self.receive_progress[x] as usize],
                            container
                        );
                        self.receive_progress[x] += 1;
                        self.receive_turn[x] = (turn + 1) as u8;
                        if self.receive_progress[x] as usize == GRID_SIZE {
                            // receive the last container in a receiving gate
                            self.pop_stack(turn, pos_bit);
                            self.grid[pick_up_pos] = NUM_CONTAINERS as u8;
                        } else {
                            // next container becomes available for receive
                            self.grid[pick_up_pos] =
                                env.receive_order[x][self.receive_progress[x] as usize] as u8;
                        }
                    } else {
                        // pick up a stacked container
                        self.pop_stack(turn, pos_bit);
                        self.grid[pick_up_pos] = NUM_CONTAINERS as u8;
                    }
                }
                Action::Release => {
                    debug_assert_eq!(release_pos, NUM_CONTAINERS);
                    release_pos = (pos_bit as u32).trailing_zeros() as usize;
                    debug_assert_ne!(container, NUM_CONTAINERS);
                    debug_assert_eq!(self.grid[release_pos] as usize, NUM_CONTAINERS);
                    if release_pos % GRID_SIZE == GRID_SIZE - 1 {
                        // dispatch
                        let x = release_pos / GRID_SIZE;
                        debug_assert_eq!(x, container / GRID_SIZE);
                        debug_assert_eq!(self.dispatch_progress[x] as usize, container);
                        self.dispatch_progress[x] += 1;
                        self.dispatch_turn[x] = (turn + 1) as u8;
                    } else {
                        // stack
                        self.push_stack(turn, pos_bit);
                        self.grid[release_pos] = container as u8;
                        self.stack_turn[release_pos] = turn as u8;
                    }
                }
                _ => {
                    pos_bit = move_pos_bit(pos_bit, *action);
                }
            };
            debug_assert_eq!(self.used_by_crane[turn] & pos_bit, 0);
            self.used_by_crane[turn] |= pos_bit;
            if Action::Up <= *action && *action <= Action::Right {
                debug_assert_ne!(self.masks[turn - 1][*action as usize ^ 1] & pos_bit, 0);
                self.masks[turn - 1][*action as usize ^ 1] &= !pos_bit;
            }
            if crane > 0 && container != NUM_CONTAINERS {
                // cannot traverse container
                self.last_turn[(pos_bit as u32).trailing_zeros() as usize].chmax((turn + 1) as u8);
            }
        }
        self.zobrist_hash = self.get_new_hash(pick_up_pos, release_pos, container, env);
        self.max_actions_len
            .chmax(self.actions[crane].len() + directions.len());
        self.sum_actions_len += directions.len();
        self.stack_cost = self.calc_new_stack_cost(container, pick_up_pos, release_pos);
        self.crane_pos_bonus = self.crane_pos_bonus + self.crane_pos[crane] as usize % GRID_SIZE
            - release_pos % GRID_SIZE; // avoid overflow
        self.actions[crane].extend(directions);
        self.crane_pos[crane] = release_pos as u8;
        debug_assert_eq!(self.evaluate(env), candidate.cost);
    }

    fn push_stack(&mut self, turn: usize, pos_bit: u32) {
        for t in turn..self.used_by_container.len() {
            debug_assert_eq!(self.used_by_container[t] & pos_bit, 0);
            self.used_by_container[t] |= pos_bit;
        }
        debug_assert_eq!(self.stack_pos & pos_bit, 0);
        self.stack_pos |= pos_bit;
    }

    fn pop_stack(&mut self, turn: usize, pos_bit: u32) {
        for t in turn..self.used_by_container.len() {
            debug_assert_ne!(self.used_by_container[t] & pos_bit, 0);
            self.used_by_container[t] &= !pos_bit;
        }
        debug_assert_ne!(self.stack_pos & pos_bit, 0);
        self.stack_pos &= !pos_bit;
    }
}

#[derive(Clone, Default)]
struct Candidate {
    cost: u16,
    state_id: u16,
    crane: u8,
    num_directions: u8,
    directions_id: u32,
    candidate_id: u16,
}

struct Selector {
    n: usize,
    log: usize,
    size: usize,
    segment_tree: Vec<Candidate>,
    p: usize,
    hash_to_index: HashMap<u64, u16>,
}

impl Selector {
    fn new(capacity: usize) -> Selector {
        let n = capacity;
        let log = 64 - n.saturating_sub(1).leading_zeros() as usize;
        let size = 1 << log;
        Selector {
            n,
            log,
            size,
            segment_tree: vec![Candidate::default(); 2 * size],
            p: size,
            hash_to_index: HashMap::with_capacity(capacity),
        }
    }

    fn accept_cost(&self, cost: u16) -> bool {
        self.p < self.size + self.n || self.segment_tree[1].cost > cost
    }

    fn accept_hash(&self, cost: u16, hash: u64) -> bool {
        if let Some(&p) = self.hash_to_index.get(&hash) {
            self.segment_tree[p as usize].cost > cost
        } else {
            true
        }
    }

    fn push(&mut self, mut candidate: Candidate, hash: u64) {
        if let Some(&p) = self.hash_to_index.get(&hash) {
            debug_assert!(self.segment_tree[p as usize].cost > candidate.cost);
            candidate.candidate_id = p;
            self.segment_tree[p as usize] = candidate;
            for i in 1..=self.log {
                self.update((p >> i) as usize);
            }
        } else if self.p == self.size + self.n {
            // full
            debug_assert!(self.segment_tree[1].cost > candidate.cost);
            let p = self.segment_tree[1].candidate_id as usize;
            candidate.candidate_id = p as u16;
            self.segment_tree[p] = candidate;
            for i in 1..=self.log {
                self.update(p >> i);
            }
            self.hash_to_index.insert(hash, p as u16);
        } else {
            // not full
            candidate.candidate_id = self.p as u16;
            self.segment_tree[self.p] = candidate;
            self.hash_to_index.insert(hash, self.p as u16);
            self.p += 1;
            if self.p == self.size + self.n {
                // become full
                // build segment tree
                for i in (1..self.size).rev() {
                    self.update(i);
                }
            }
        }
    }

    fn select(&self) -> Vec<Candidate> {
        let mut ret = self.segment_tree[self.size..self.p].to_vec();
        ret.sort_by_key(|candidate| candidate.cost);
        ret
    }

    fn clear(&mut self) {
        self.p = self.size;
        self.hash_to_index.clear();
        self.segment_tree.fill(Candidate::default());
    }

    fn update(&mut self, k: usize) {
        self.segment_tree[k] = if self.segment_tree[2 * k].cost >= self.segment_tree[2 * k + 1].cost
        {
            self.segment_tree[2 * k].clone()
        } else {
            self.segment_tree[2 * k + 1].clone()
        }
    }
}

fn move_pos_bit(pos_bit: u32, direction: Action) -> u32 {
    match direction {
        Action::Up => pos_bit >> GRID_SIZE,
        Action::Down => pos_bit << GRID_SIZE,
        Action::Left => pos_bit >> 1,
        Action::Right => pos_bit << 1,
        _ => pos_bit,
    }
}

fn move_pos_bit_reverse(pos_bit: u32, direction: Action) -> u32 {
    match direction {
        Action::Up => pos_bit << GRID_SIZE,
        Action::Down => pos_bit >> GRID_SIZE,
        Action::Left => pos_bit << 1,
        Action::Right => pos_bit >> 1,
        _ => pos_bit,
    }
}

fn bit_set(s: &mut u32, pos: usize) {
    debug_assert!(pos < NUM_CONTAINERS);
    *s |= 1 << pos;
}

// fn bit_reset(s: &mut u32, pos: usize) {
//     debug_assert!(pos < NUM_CONTAINERS);
//     *s &= !(1 << pos);
// }

fn bit_get(s: u32, pos: usize) -> bool {
    debug_assert!(pos < NUM_CONTAINERS);
    (s & (1 << pos)) > 0
}

fn manhattan(sx: usize, sy: usize, tx: usize, ty: usize) -> usize {
    sx.abs_diff(tx) + sy.abs_diff(ty)
}

pub fn get_time_sec() -> f64 {
    static mut STIME: f64 = -1.0;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        ms - STIME
    }
}
