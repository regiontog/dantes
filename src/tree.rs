use ndarray::Array1;

use super::{Perspective, Policy};
use mcts::Action;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct NodeId {
    index: usize,
}

pub(crate) struct Node {
    prior_probabilities: Array1<f64>,
    scores: Array1<f64>,
    visits: Array1<f64>,
    children: Option<Vec<NodeId>>,
    parent: Option<NodeId>,
    index_in_parent: usize,
}

impl Node {
    fn new(max_children: usize, parent: Option<NodeId>, index_in_parent: usize) -> Self {
        Node {
            prior_probabilities: Array1::zeros((max_children,)),
            scores: Array1::zeros((max_children,)),
            visits: Array1::zeros((max_children,)),
            children: None,
            index_in_parent,
            parent,
        }
    }

    fn expand(&mut self, children: Vec<NodeId>) {
        debug_assert!(self.children.is_none());

        self.children = Some(children);
    }

    pub(crate) fn is_expanded(&self) -> bool {
        self.children.is_some()
    }

    fn best_child<A: Action>(
        &self,
        perspective: Perspective,
        self_visits: f64,
        policy: impl Policy,
        actions: &[A],
    ) -> (NodeId, A) {
        let action = perspective
            .minmax_arg(policy.evaluate(
                &self.prior_probabilities,
                &self.scores,
                &self.visits,
                self_visits,
            )).expect("There to be more than 0 possible child states!");

        (
            self.children
                .as_ref()
                .expect("Cannot find best child of unexpanded node!")[action],
            actions[action],
        )
    }
}

pub(crate) struct Arena {
    root_visits: f64,
    root_score: f64,
    root_id: NodeId,
    nodes: Vec<Node>,
    max_children: usize,
}

impl Arena {
    pub(crate) fn with_root(max_children: usize) -> Self {
        let this = Arena {
            root_visits: 0.,
            root_score: 0.,
            root_id: NodeId { index: 0 },
            nodes: vec![],
            max_children,
        };

        this.create_root();
        this
    }

    pub(crate) fn get_root_id(&self) -> NodeId {
        self.root_id
    }

    fn create_node(&mut self, parent: Option<NodeId>, index_in_parent: usize) -> NodeId {
        self.nodes
            .push(Node::new(self.max_children, parent, index_in_parent));

        NodeId {
            index: self.nodes.len(),
        }
    }

    fn create_root(&mut self) -> NodeId {
        self.nodes.truncate(0);
        self.root_id = self.create_node(None, 0);
        self.root_score = 0.;
        self.root_visits = 0.;
        self.root_id
    }

    fn expand_node(&mut self, node: NodeId, num_childs: usize) {
        debug_assert!(num_childs < self.max_children);

        let children = (0..num_childs)
            .map(|i| self.create_node(Some(node), i))
            .collect();

        self.nodes[node.index].expand(children);
    }

    pub(crate) fn borrow(&self, node: NodeId) -> &Node {
        &self.nodes[node.index]
    }

    pub(crate) fn best_child<A: Action>(
        &self,
        perspective: Perspective,
        node: NodeId,
        policy: impl Policy,
        actions: &[A],
    ) -> (NodeId, A) {
        self.nodes[node.index].best_child(perspective, self.visits(node), policy, actions)
    }

    fn visits(&self, node: NodeId) -> f64 {
        if node == self.root_id {
            self.root_visits
        } else {
            let n = &self.nodes[node.index];
            let parent = &self.nodes[n
                                         .parent
                                         .expect("All non-root nodes should have a parent!")
                                         .index];
            parent.visits[n.index_in_parent]
        }
    }

    fn visit(&mut self, node: NodeId) {
        if node == self.root_id {
            self.root_visits += 1.
        } else {
            let parent_id = self.nodes[node.index].parent;
            let index = self.nodes[node.index].index_in_parent;

            let parent = &mut self.nodes[parent_id
                                             .expect("All non-root nodes should have a parent!")
                                             .index];

            parent.visits[index] += 1.
        }
    }

    fn add_score(&mut self, node: NodeId, score: f64) {
        if node == self.root_id {
            self.root_score += score
        } else {
            let parent_id = self.nodes[node.index].parent;
            let index = self.nodes[node.index].index_in_parent;

            let parent = &mut self.nodes[parent_id
                                             .expect("All non-root nodes should have a parent!")
                                             .index];

            parent.scores[index] += score
        }
    }
}
