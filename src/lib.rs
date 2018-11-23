#[macro_use]
extern crate ndarray;
extern crate ndarray_npy;
extern crate num_integer;
extern crate rand;
extern crate rayon;
extern crate tensorflow;

use ndarray::Array1;

pub mod ai;
pub mod evaluators;
pub mod mcts;
pub mod policies;
mod tree;

pub trait Action: Copy {}

impl<C: Copy> Action for C {}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Player {
    Player1,
    Player2,
}

impl Player {
    pub fn swap(self) -> Self {
        match self {
            Player::Player1 => Player::Player2,
            Player::Player2 => Player::Player1,
        }
    }
}

pub type FullState<G> = (Player, <G as Game>::GameState);

#[derive(Debug, PartialEq, Eq)]
pub enum GameResult {
    Tie,
    Won(Player),
    Ongoing,
}

pub trait Game {
    type GameState: Clone;
    type GameAction: Action + Sized;

    fn initial_state(&self) -> FullState<Self>;
    fn max_child_states(&self) -> usize;
    fn take_action(&self, &FullState<Self>, Self::GameAction) -> FullState<Self>;
    fn num_actions(&self, &FullState<Self>) -> usize;
    fn with_action_space<R>(&self, &FullState<Self>, impl Fn(&[Self::GameAction]) -> R) -> R;
    fn result(&self, &FullState<Self>) -> GameResult;
}

#[derive(Copy, Clone)]
pub enum Perspective {
    PreferPositive,
    PreferNegative,
}

impl Perspective {
    pub fn maybe_negate<F>(self, array: Array1<F>) -> Array1<F>
    where
        F: ::std::ops::Neg<Output = F>,
        F: Copy,
    {
        match self {
            Perspective::PreferPositive => array,
            Perspective::PreferNegative => -array,
        }
    }
}
