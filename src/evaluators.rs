use ndarray::Array1;

use super::{FullState, Game};
use policies::{Policy, Random};

pub trait StateEvaluator<G: Game> {
    type TargetPolicy: Policy;

    fn evaluate(&self, &FullState<G>) -> Array1<f64>;
}

pub struct RandomEvaluator<G: Game> {
    game: G,
}

impl<G: Game> RandomEvaluator<G> {
    pub fn new(game: G) -> Self {
        RandomEvaluator { game }
    }
}

impl<G: Game> StateEvaluator<G> for RandomEvaluator<G> {
    type TargetPolicy = Random;

    fn evaluate(&self, state: &FullState<G>) -> Array1<f64> {
        let num_actions = self.game.num_actions(state);
        Array1::ones((num_actions,))
    }
}
