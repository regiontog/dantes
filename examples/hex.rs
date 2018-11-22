extern crate chrono;
extern crate dantes;
extern crate ndarray;
extern crate rand;
extern crate tensorflow;

use std::collections::HashMap;
use std::collections::HashSet;

use chrono::{DateTime, Local};
use ndarray::Array1;
use rand::seq::{IteratorRandom, SliceRandom};
use tensorflow::Tensor;

use dantes::ai::{BestProbability, TensorflowConverter, TensorflowEvaluator, UCTAI};
use dantes::mcts::MonteCarloTreeSearch;
use dantes::policies::{BestQuality, MinMax, WeightedRandom};
use dantes::{FullState, Game, GameResult, Player};

struct Hex {
    full_action_space: Box<[<Hex as Game>::GameAction]>,
    board_size: u8,
    neighbors: HashMap<<Hex as Game>::GameAction, Vec<<Hex as Game>::GameAction>>,
    walls: Box<[Box<Fn(u8, u8) -> Option<Wall> + Sync + Send>]>,
    action_index: HashMap<<Hex as Game>::GameAction, usize>,
}

// const NEIGHBORS: [(i16, i16); 6] = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)];
const NEIGHBORS: [(i16, i16); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)];

impl Clone for Hex {
    fn clone(&self) -> Hex {
        Hex {
            board_size: self.board_size,
            neighbors: self.neighbors.clone(),
            action_index: self.action_index.clone(),
            full_action_space: self.full_action_space.clone(),
            walls: Self::walls(self.board_size),
        }
    }
}

impl Hex {
    fn describe_move(state: &FullState<Self>, action: <Self as Game>::GameAction) {
        println!("{:?} sets down stone on {:?}", state.0.swap(), action);
    }

    fn new(n: u8) -> Self {
        let mut full_action_space = Vec::with_capacity(n as usize * n as usize);
        let mut neighbors = HashMap::new();

        for cell in (0..n).flat_map(|i| (0..n).map(move |j| (i, j))) {
            full_action_space.push(cell);
            neighbors.insert(cell, Self::neighbors(cell, n));
        }

        Self {
            neighbors,
            board_size: n,
            action_index: full_action_space
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, action)| (action, i))
                .collect(),
            full_action_space: full_action_space.into_boxed_slice(),
            walls: Self::walls(n),
        }
    }

    fn range_check(action: (i16, i16), n: u8) -> Option<<Self as Game>::GameAction> {
        if action.0 >= 0 && action.0 < n as i16 && action.1 >= 0 && action.1 < n as i16 {
            Some((action.0 as u8, action.1 as u8))
        } else {
            None
        }
    }

    fn neighbors(action: <Self as Game>::GameAction, n: u8) -> Vec<<Self as Game>::GameAction> {
        NEIGHBORS
            .iter()
            .map(|offset| (action.0 as i16 + offset.0, action.1 as i16 + offset.1))
            .filter_map(|action| Self::range_check(action as (i16, i16), n))
            .collect()
    }

    fn walls(n: u8) -> Box<[Box<Fn(u8, u8) -> Option<Wall> + Sync + Send>]> {
        Box::new([
            Box::new(move |r, _| if r == 0 { Some(Wall::TopRight) } else { None }),
            Box::new(move |r, _| {
                if r == n - 1 {
                    Some(Wall::BottomLeft)
                } else {
                    None
                }
            }),
            Box::new(move |_, c| if c == 0 { Some(Wall::TopLeft) } else { None }),
            Box::new(move |_, c| {
                if c == n - 1 {
                    Some(Wall::BottomRight)
                } else {
                    None
                }
            }),
        ])
    }

    fn wall<'a>(&'a self, action: <Self as Game>::GameAction) -> impl Iterator<Item = Wall> + 'a {
        self.walls.iter().filter_map(move |f| f(action.0, action.1))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum CellState {
    Empty,
    Occupied(Player),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Wall {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

// TODO: lru_cache for results
#[derive(Clone)]
struct HexState {
    board: Box<[Box<[CellState]>]>,
    prev_action: Option<<Hex as Game>::GameAction>,
    move_number: usize,
}

impl std::fmt::Display for HexState {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        // write!(fmt, "{:?}\n", self.prev_action)?;
        // write!(fmt, "{}\n", self.move_number)?;

        let n = self.board.len();

        let midline = n * 2 + 1;

        let cells: Vec<_> = (0..n)
            .flat_map(|i| (0..=i).map(move |j| (i - j, j)))
            .chain((1..n).flat_map(|j| (j..n).rev().map(move |i| (i, n - i + j - 1))))
            .collect();

        let mut cell_index = 0;

        let color = |cell: (usize, usize)| match self.board[cell.0 as usize][cell.1 as usize] {
            CellState::Empty => " ",
            CellState::Occupied(player) => if player == Player::Player1 {
                "r"
            } else {
                "b"
            },
        };

        let top = vec![|cell| format!("/{}\\", color(cell)); n];
        let bot = vec!["\\_/"; n];

        write!(fmt, "{}_", str::repeat(" ", midline))?;
        for i in 1..n {
            write!(
                fmt,
                "\n{}_{}_",
                str::repeat(" ", midline - 2 * i),
                top[..i]
                    .iter()
                    .map(|f| {
                        let r = f(cells[cell_index]);
                        cell_index += 1;
                        r
                    }).collect::<Vec<_>>()
                    .join("_")
            )?;
        }
        write!(
            fmt,
            "\n{}{}",
            str::repeat(" ", midline - 2 * n + 1),
            top[..n]
                .iter()
                .map(|f| {
                    let r = f(cells[cell_index]);
                    cell_index += 1;
                    r
                }).collect::<Vec<_>>()
                .join("_")
        )?;

        for i in (1..=n).rev() {
            write!(
                fmt,
                "\n{}{}",
                str::repeat(" ", midline - 2 * i + 1),
                bot[..i - 1]
                    .iter()
                    .map(|bot| {
                        let r = format!("{}{}", bot, color(cells[cell_index]));
                        cell_index += 1;
                        r
                    }).chain(std::iter::once(format!("{}", bot[i - 1])))
                    .collect::<Vec<_>>()
                    .join("")
            )?;
        }

        Ok(())
    }
}

impl Game for Hex {
    type GameState = HexState;
    type GameAction = (u8, u8);

    fn initial_state(&self) -> FullState<Self> {
        let empty_row = (0..self.board_size)
            .map(|_| CellState::Empty)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        (
            Player::Player1,
            HexState {
                board: (0..self.board_size)
                    .map(|_| empty_row.clone())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                move_number: 0,
                prev_action: None,
            },
        )
    }

    fn max_child_states(&self) -> usize {
        self.board_size as usize * self.board_size as usize
    }

    fn num_actions(&self, state: &FullState<Self>) -> usize {
        if let GameResult::Won(_) = self.result(state) {
            0
        } else {
            self.board_size as usize * self.board_size as usize - state.1.move_number
        }
    }

    fn take_action(&self, state: &FullState<Self>, action: Self::GameAction) -> FullState<Self> {
        let (player, state) = state;
        let mut new_state = state.clone();

        let (i, j) = action;

        new_state.board[i as usize][j as usize] = CellState::Occupied(*player);
        new_state.prev_action = Some(action);
        new_state.move_number += 1;

        (player.swap(), new_state)
    }

    fn with_action_space<R>(
        &self,
        state: &FullState<Self>,
        continuation: impl Fn(&[Self::GameAction]) -> R,
    ) -> R {
        let (_, state) = state;

        continuation(
            &self
                .full_action_space
                .iter()
                .cloned()
                .filter(|action| {
                    let (i, j) = action;

                    state.board[*i as usize][*j as usize] == CellState::Empty
                }).collect::<Vec<_>>()[..],
        )
    }

    fn result(&self, state: &FullState<Self>) -> GameResult {
        let (next_player, hexstate) = state;

        let (mut left_wall, mut right_wall) = (false, false);

        let maybe_winner = next_player.swap();

        hexstate
            .prev_action
            .map(|prev_action| {
                let mut connected_moves = vec![prev_action];
                let mut visited = HashSet::new();

                while let Some((i, j)) = connected_moves.pop() {
                    visited.insert((i, j));

                    match hexstate.board[i as usize][j as usize] {
                        CellState::Occupied(occupied_by) if occupied_by == maybe_winner => {
                            for wall in self.wall((i, j)) {
                                match wall {
                                    Wall::TopLeft => if maybe_winner == Player::Player2 {
                                        left_wall = true;
                                    },
                                    Wall::BottomRight => if maybe_winner == Player::Player2 {
                                        right_wall = true;
                                    },
                                    Wall::TopRight => if maybe_winner == Player::Player1 {
                                        right_wall = true;
                                    },
                                    Wall::BottomLeft => if maybe_winner == Player::Player1 {
                                        left_wall = true;
                                    },
                                }
                            }

                            if left_wall && right_wall {
                                return GameResult::Won(maybe_winner);
                            }

                            connected_moves.extend(
                                self.neighbors[&(i, j)]
                                    .iter()
                                    .filter(|a| !visited.contains(a)),
                            );
                        }
                        _ => {}
                    }
                }

                return GameResult::Ongoing;
            }).unwrap_or(GameResult::Ongoing)
    }
}

impl TensorflowConverter for Hex {
    type D = Tensor<f64>;

    fn nn_state_representation(&self, state: &FullState<Self>) -> Tensor<f64> {
        let repr = std::iter::once(match state.0 {
            Player::Player1 => 1.,
            Player::Player2 => 2.,
        }).chain(
            state
                .1
                .board
                .iter()
                .flat_map(|col| col.iter())
                .map(|cell| match cell {
                    CellState::Empty => 0.,
                    CellState::Occupied(Player::Player1) => 1.,
                    CellState::Occupied(Player::Player2) => 2.,
                }),
        ).collect::<Vec<_>>();

        debug_assert!(repr.len() == self.max_child_states() + 1);

        Tensor::new(&[1, self.max_child_states() as u64 + 1])
            .with_values(&repr)
            .unwrap()
    }

    fn to_distribution(&self, state: &FullState<Self>, y: Tensor<f64>) -> Array1<f64> {
        self.with_action_space(state, |actions| {
            let legal_action_dist =
                Array1::from_iter(actions.iter().map(|action| y[self.action_index[action]]));

            let norm = f64::sqrt(legal_action_dist.mapv(|v| v.powi(2)).scalar_sum());

            legal_action_dist / norm
        })
    }

    fn normalize_distribution(
        &self,
        state: &FullState<Self>,
        mut distribution: Array1<f64>,
    ) -> Self::D {
        let min = *distribution
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Greater))
            .unwrap_or(&0.);

        distribution = distribution.mapv(|v| v - min);

        let norm = f64::sqrt(distribution.mapv(|v| v.powi(2)).scalar_sum());
        distribution = distribution / norm;

        // Set all illegal moves to 0
        distribution = self.with_action_space(state, |actions| {
            let mut full_distribution = Array1::zeros((self.max_child_states(),));

            for (i, action) in actions.iter().enumerate() {
                full_distribution[self.action_index[action]] = distribution[i];
            }

            full_distribution
        });

        debug_assert!(distribution.len() == self.max_child_states());

        Tensor::new(&[distribution.len() as u64])
            .with_values(distribution.as_slice().unwrap())
            .unwrap()
    }

    fn select(
        &self,
        training_set: impl Iterator<Item = (Tensor<f64>, Self::D)>,
    ) -> (Tensor<f64>, Tensor<f64>) {
        let mut rng = rand::thread_rng();

        let mut batch = training_set.choose_multiple(&mut rng, BATCH_SIZE);
        batch.shuffle(&mut rng);

        let actual_batch_size = batch.len();

        let mut xs: Vec<f64> =
            Vec::with_capacity(actual_batch_size * (self.max_child_states() + 1));
        let mut ys: Vec<f64> = Vec::with_capacity(actual_batch_size * self.max_child_states());

        for (x, y) in batch.iter() {
            xs.extend(x.iter());
            ys.extend(y.iter());
        }

        let tx = Tensor::new(&[actual_batch_size as u64, self.max_child_states() as u64 + 1])
            .with_values(&xs)
            .unwrap();

        let ty = Tensor::new(&[actual_batch_size as u64, self.max_child_states() as u64])
            .with_values(&ys)
            .unwrap();

        (tx, ty)
    }
}

const BATCH_SIZE: usize = 100;

fn main() {
    let m = 100;
    let i = 10;

    let start_date: DateTime<Local> = Local::now();

    let hex = Hex::new(5);

    let ai = TensorflowEvaluator::<_, BestProbability<WeightedRandom>>::load_model_from_file(
        hex.clone(),
        "examples/net/model.pb",
    ).unwrap();

    let mut mcts = MonteCarloTreeSearch::new(hex.clone(), ai);

    for game_num in 0..m {
        let mut game = hex.initial_state();
        let mut replay_buffer = vec![];

        while let GameResult::Ongoing = hex.result(&game) {
            let tree = mcts.par_simulate::<UCTAI<WeightedRandom>>(&game, 100, 10);
            let d = mcts.distribution::<BestQuality<MinMax>>(&tree, &game);

            replay_buffer.push((game.clone(), d.clone()));

            match mcts.best_action::<MinMax>(&game, &d) {
                None => unreachable!(),
                Some(action) => {
                    game = hex.take_action(&game, action);
                    Hex::describe_move(&game, action);
                    println!("{}\n", game.1);
                }
            }
        }

        println!("Game over! Winner is: {:?}", hex.result(&game));

        mcts.mut_evaluator().train(replay_buffer);

        if game_num % i == 0 {
            mcts.mut_evaluator()
                .save(
                    &format!("examples/net/{:?}-{}", start_date, game_num),
                    &[
                        "fully_connected/weights:0",
                        "fully_connected/bias:0",
                        "fully_connected_1/weights:0",
                        "fully_connected_1/bias:0",
                        "fully_connected_2/weights:0",
                        "fully_connected_2/bias:0",
                        "fully_connected_3/weights:0",
                        "fully_connected_3/bias:0",
                        "y_/weights:0",
                        "y_/bias:0",
                    ],
                ).unwrap();
        }

        println!("Trained net");
    }
}
