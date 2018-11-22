use std::sync::RwLock;

use ndarray::Array1;
use num_integer::Integer;
use rayon::prelude::*;

use super::{FullState, Game, GameResult, Perspective, Player};
use evaluators::{RandomEvaluator, StateEvaluator};
use policies::{Policy, SelectionStrategy};
use tree::{Arena, NodeId};

pub struct MonteCarloTreeSearch<G: Game, E: StateEvaluator<G>> {
    game: G,
    evaluator: E,
}

impl<G: Game> MonteCarloTreeSearch<G, RandomEvaluator<G>> {
    pub fn random_rollout(game: &G) -> Self
    where
        G: Clone,
    {
        MonteCarloTreeSearch {
            game: game.clone(),
            evaluator: RandomEvaluator::new(game.clone()),
        }
    }
}

impl<G: Game, E: StateEvaluator<G>> MonteCarloTreeSearch<G, E> {
    pub fn new(game: G, evaluator: E) -> Self {
        MonteCarloTreeSearch { game, evaluator }
    }

    fn perspective_for(player: Player) -> Perspective {
        match player {
            Player::Player1 => Perspective::PreferPositive,
            Player::Player2 => Perspective::PreferNegative,
        }
    }

    fn select<TreePolicy: Policy>(
        &self,
        tree: &Arena,
        state: &FullState<G>,
    ) -> (NodeId, FullState<G>) {
        let mut cur_node = tree.get_root_id();
        let mut node = tree.borrow(cur_node);

        if !node.is_expanded() || !node.has_children() {
            return (cur_node, state.clone());
        }

        let (player, _) = state;

        let (best_node, action) = self.game.with_action_space(&state, |actions| {
            tree.best_child::<TreePolicy, _>(Self::perspective_for(*player), cur_node, actions)
        });

        cur_node = best_node;
        node = tree.borrow(cur_node);

        let mut state = self.game.take_action(state, action);

        while node.is_expanded() {
            let (player, _) = state;

            if !node.has_children() {
                return (cur_node, state.clone());
            }

            let (best_node, action) = self.game.with_action_space(&state, |actions| {
                tree.best_child::<TreePolicy, _>(Self::perspective_for(player), cur_node, actions)
            });

            cur_node = best_node;
            node = tree.borrow(cur_node);
            state = self.game.take_action(&state, action);
        }

        (cur_node, state)
    }

    fn rollout<TargetPolicy: Policy>(&self, mut state: FullState<G>) -> (f64, Array1<f64>) {
        let prior_distribution = self.evaluator.evaluate(&state);
        let mut running_distribution = prior_distribution.clone();

        let zeros = Array1::zeros((self.game.max_child_states(),));

        while let GameResult::Ongoing = self.game.result(&state) {
            let action = self.game.with_action_space(&state, |actions| {
                // TODO: Clean probabilities based on which action are available and normalize.
                let (player, _) = state;
                let num_actions = self.game.num_actions(&state);

                let index = TargetPolicy::evaluate(
                    Self::perspective_for(player),
                    &running_distribution.slice(s![..num_actions]),
                    &zeros.slice(s![..num_actions]),
                    &zeros.slice(s![..num_actions]),
                    0.,
                ).unwrap_or_else(|| {
                    println!("");
                    println!("{}", num_actions);
                    println!("{}", running_distribution);
                    println!("");
                    panic!("There to always be an action to take during rollout.")
                });

                actions[index]
            });

            state = self.game.take_action(&state, action);
            running_distribution = self.evaluator.evaluate(&state);
        }

        (
            match self.game.result(&state) {
                GameResult::Ongoing => unreachable!(),
                GameResult::Tie => 0.,
                GameResult::Won(winner) => if winner == Player::Player1 {
                    1.
                } else {
                    -1.
                },
            },
            prior_distribution,
        )
    }

    fn backpropogate(tree: &mut Arena, mut node: NodeId, reward: f64) {
        tree.visit(node);
        tree.add_score(node, reward);

        while let Some(parent) = tree.borrow(node).parent {
            node = parent;

            tree.visit(node);
            tree.add_score(node, reward);
        }
    }

    pub fn par_simulate<TreePolicy: Policy>(
        &mut self,
        state: &FullState<G>,
        num_simulations: u64,
        par_iter: u64,
    ) -> Arena
    where
        G::GameState: Sync,
        G: Sync + Send,
        E: Sync + Send,
    {
        let mut tree = Arena::with_root(self.game.max_child_states());

        for chunk_size in chunk_sizes(num_simulations, par_iter) {
            let rollouts: Vec<_> = (0..chunk_size)
                .into_par_iter()
                .map(|_| {
                    let (leaf, leaf_state) = self.select::<TreePolicy>(&tree, &state);
                    let num_childs = self.game.num_actions(&leaf_state);

                    (
                        leaf,
                        num_childs,
                        self.rollout::<E::TargetPolicy>(leaf_state),
                    )
                }).collect();

            for (leaf, num_childs, (reward, prior_probabilities)) in rollouts {
                if !tree.borrow(leaf).is_expanded() {
                    // In case we end up in a situation where the selected node has an empty action space
                    // Then we still want to backprop so that the next select has a shot at selecting an unexpanded node.
                    tree.expand_node(leaf, num_childs);
                    tree.set_prior_probabilities(leaf, prior_probabilities);
                }

                Self::backpropogate(&mut tree, leaf, reward);
            }
        }

        tree
    }

    pub fn par_search<TreePolicy: Policy, SearchPolicy: Policy>(
        &mut self,
        state: &FullState<G>,
        num_simulations: u64,
        par_iter: u64,
    ) -> Option<G::GameAction>
    where
        G::GameState: Sync,
        G: Sync + Send,
        E: Sync + Send,
    {
        let tree = self.par_simulate::<TreePolicy>(state, num_simulations, par_iter);
        let (root_player, _) = state;

        if self.game.num_actions(&state) > 0 {
            self.game.with_action_space(&state, |actions| {
                println!("{:?}", tree.borrow(tree.get_root_id()));

                let (_, action) = tree.best_child::<SearchPolicy, _>(
                    Self::perspective_for(*root_player),
                    tree.get_root_id(),
                    actions,
                );

                Some(action)
            })
        } else {
            None
        }
    }

    pub fn par_search_locked<TreePolicy: Policy, SearchPolicy: Policy>(
        &mut self,
        state: FullState<G>,
        num_simulations: u64,
    ) -> Option<G::GameAction>
    where
        G::GameState: Sync,
        G: Sync + Send,
        E: Sync + Send,
    {
        let mut tree = Arena::with_root(self.game.max_child_states());
        let (root_player, _) = state;

        {
            let locked_tree = RwLock::new(&mut tree);

            (0..num_simulations).into_par_iter().for_each(|_| {
                let ((reward, pp), exp, leaf, num) = match locked_tree.read() {
                    Err(_) => panic!("Poisoned lock!"),
                    Ok(tree) => {
                        let (leaf, leaf_state) = self.select::<TreePolicy>(&tree, &state);

                        let num_childs = self.game.num_actions(&leaf_state);

                        (
                            self.rollout::<E::TargetPolicy>(leaf_state),
                            tree.borrow(leaf).is_expanded(),
                            leaf,
                            num_childs,
                        )
                    }
                };

                match locked_tree.write() {
                    Err(_) => panic!("Poisoned lock!"),
                    Ok(mut tree) => {
                        if !exp {
                            // In case we end up in a situation where the selected node has an empty action space
                            // Then we still want to backprop so that the next select has a shot at selecting an unexpanded node.
                            tree.expand_node(leaf, num);
                            tree.set_prior_probabilities(leaf, pp);
                        }

                        Self::backpropogate(&mut tree, leaf, reward);
                    }
                };
            });
        }

        if self.game.num_actions(&state) > 0 {
            self.game.with_action_space(&state, |actions| {
                // println!("{:?}", tree.borrow(tree.get_root_id()));

                let (_, action) = tree.best_child::<SearchPolicy, _>(
                    Self::perspective_for(root_player),
                    tree.get_root_id(),
                    actions,
                );

                Some(action)
            })
        } else {
            None
        }
    }

    pub fn search<TreePolicy: Policy, SearchPolicy: Policy>(
        &mut self,
        state: &FullState<G>,
        num_simulations: u64,
    ) -> Option<G::GameAction>
    where
        G::GameState: Sync,
        G: Sync + Send,
        E: Sync + Send,
    {
        let mut tree = Arena::with_root(self.game.max_child_states());
        let (root_player, _) = state;

        for _ in 0..num_simulations {
            let (leaf, leaf_state) = self.select::<TreePolicy>(&tree, &state);
            let num_childs = self.game.num_actions(&leaf_state);

            let (reward, prior_probabilities) = self.rollout::<E::TargetPolicy>(leaf_state);

            if !tree.borrow(leaf).is_expanded() {
                // In case we end up in a situation where the selected node has an empty action space
                // Then we still want to backprop so that the next select has a shot at selecting an unexpanded node.
                tree.expand_node(leaf, num_childs);
                tree.set_prior_probabilities(leaf, prior_probabilities);
            }

            Self::backpropogate(&mut tree, leaf, reward);
        }

        if self.game.num_actions(&state) > 0 {
            self.game.with_action_space(&state, |actions| {
                println!("{:?}", tree.borrow(tree.get_root_id()));

                let (_, action) = tree.best_child::<SearchPolicy, _>(
                    Self::perspective_for(*root_player),
                    tree.get_root_id(),
                    actions,
                );

                Some(action)
            })
        } else {
            None
        }
    }

    pub fn distribution<P: Policy>(&self, tree: &Arena, state: &FullState<G>) -> Array1<f64> {
        let (player, _) = state;

        tree.distribution::<P>(Self::perspective_for(*player), tree.get_root_id())
    }

    pub fn best_action<S: SelectionStrategy>(
        &self,
        state: &FullState<G>,
        distribution: &Array1<f64>,
    ) -> Option<G::GameAction> {
        let (player, _) = state;

        self.game.with_action_space(state, |actions| {
            S::select(Self::perspective_for(*player), distribution).map(|i| actions[i])
        })
    }

    pub fn mut_evaluator(&mut self) -> &mut E {
        &mut self.evaluator
    }
}

fn chunk_sizes(total_iters: u64, max_chunk_size: u64) -> impl Iterator<Item = u64> {
    let (quotient, remainder) = total_iters.div_rem(&max_chunk_size);

    ChunkSizesIterator {
        range: 0..quotient,
        repeat: max_chunk_size,
        then: if remainder == 0 {
            None
        } else {
            Some(remainder)
        },
    }
}

struct ChunkSizesIterator {
    range: std::ops::Range<u64>,
    then: Option<u64>,
    repeat: u64,
}

impl Iterator for ChunkSizesIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.range
            .next()
            .map(|_| self.repeat)
            .or_else(|| self.then.take())
    }
}
