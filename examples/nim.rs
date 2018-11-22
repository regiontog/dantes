extern crate dantes;

use dantes::mcts::MonteCarloTreeSearch;
use dantes::policies::{BestQuality, MinMax, WeightedRandom, UCT};
use dantes::{FullState, Game, GameResult, Player};

#[derive(Clone)]
struct Nim {
    full_action_space: Box<[u8]>,
    starting_state: usize,
    max_remove: u8,
}

impl Nim {
    fn new(n: usize, k: u8) -> Self {
        Nim {
            full_action_space: (1..=k).collect::<Vec<_>>().into_boxed_slice(),
            starting_state: n,
            max_remove: k,
        }
    }
}

impl Game for Nim {
    type GameState = usize;
    type GameAction = u8;

    fn initial_state(&self) -> FullState<Self> {
        (Player::Player1, self.starting_state)
    }

    fn max_child_states(&self) -> usize {
        self.max_remove as usize
    }

    fn num_actions(&self, state: &FullState<Self>) -> usize {
        usize::min(self.max_remove as usize, state.1)
    }

    fn take_action(&self, state: &FullState<Self>, action: Self::GameAction) -> FullState<Self> {
        (state.0.swap(), state.1 - action as usize)
    }

    fn with_action_space<R>(
        &self,
        state: &FullState<Self>,
        continuation: impl Fn(&[Self::GameAction]) -> R,
    ) -> R {
        continuation(&self.full_action_space[..self.num_actions(state)])
    }

    fn result(&self, state: &FullState<Self>) -> GameResult {
        match state.1 {
            0 => GameResult::Won(state.0.swap()),
            _ => GameResult::Ongoing,
        }
    }
}

fn main() {
    let nim = Nim::new(10, 3);
    let mut game = nim.initial_state();

    let mut mcts = MonteCarloTreeSearch::random_rollout(&nim);

    while let GameResult::Ongoing = nim.result(&game) {
        println!("{:?}", game);

        // match mcts.par_search_locked::<UCT<MinMax>, BestQuality<MinMax>>(game, 1000000) {
        match mcts.par_search::<UCT<WeightedRandom>, BestQuality<MinMax>>(&game, 1000000, 1000) {
            None => unreachable!(),
            Some(action) => {
                game = nim.take_action(&game, action);
            }
        }
    }

    println!("Game over! {:?}", nim.result(&game));
}
