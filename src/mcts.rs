use super::{Perspective, Policy};
use tree::{Arena, NodeId};

pub(crate) trait Action: Copy {}

#[derive(Copy, Clone)]
enum Player {
    Player1,
    Player2,
}

type FullState<G: Game> = (Player, G::GameState);

trait Game: Copy {
    type GameState;
    type GameAction: Action;

    fn max_child_states(self) -> usize;
    fn action_space(self, state: &FullState<Self>) -> [Self::GameAction];
    fn take_action(self, action: Self::GameAction) -> FullState<Self>;
}

struct MonteCarloTreeSearch<G: Game, P: Policy> {
    game: G,
    tree_policy: P,
}

impl<G: Game, P: Policy> MonteCarloTreeSearch<G, P> {
    fn perspective_for(player: Player) -> Perspective {
        match player {
            Player::Player1 => Perspective::PreferPositive,
            Player::Player2 => Perspective::PreferNegative,
        }
    }

    fn search(&mut self, mut state: FullState<G>) -> FullState<G> {
        let tree = Arena::with_root(self.game.max_child_states());

        // Select
        let cur_node = tree.get_root_id();

        while tree.borrow(cur_node).is_expanded() {
            let (player, _) = state;
            let actions = &self.game.action_space(&state);
            let action;

            (cur_node, action) = tree.best_child(
                Self::perspective_for(player),
                cur_node,
                self.tree_policy,
                actions,
            );

            state = self.game.take_action(action);
        }
        // Expand
        // Evaluate
        // Backpropogate

        unimplemented!()
    }
}
