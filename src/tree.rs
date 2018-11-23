use ndarray::Array1;

use super::{Action, Perspective};
use policies::Policy;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub(crate) struct NodeId {
    index: usize,
}

#[derive(Debug)]
pub(crate) struct Node {
    prior_probabilities: Array1<f64>,
    scores: Array1<f64>,
    visits: Array1<f64>,
    children: Option<Vec<NodeId>>,
    pub(crate) parent: Option<NodeId>,
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

    pub(crate) fn has_children(&self) -> bool {
        self.children
            .as_ref()
            .map(|children| children.len() > 0)
            .unwrap_or(false)
    }

    fn best_child<P: Policy, A: Action>(
        &self,
        perspective: Perspective,
        self_visits: f64,
        actions: &[A],
    ) -> (NodeId, A) {
        let num_actions = actions.len();

        debug_assert!(Some(num_actions) == self.children.as_ref().map(|c| c.len()));

        let action = P::evaluate(
            perspective,
            &self.prior_probabilities.slice(s![..num_actions]),
            &self.scores.slice(s![..num_actions]),
            &self.visits.slice(s![..num_actions]),
            self_visits,
        ).unwrap_or_else(|| {
            println!("{}", num_actions);
            println!("{:?}", self.prior_probabilities);
            println!("{:?}", self.scores);
            println!("{:?}", self.visits);
            println!("{:?}", self_visits);
            panic!("There to be more than 0 possible actions!");
        });

        (
            // Unwrap is safe so long as the expect above is clear
            self.children
                .as_ref()
                .expect("Cannot find best child of unexpanded node!")[action],
            actions[action],
        )
    }

    pub(crate) fn distribution<P: Policy>(
        &self,
        perspective: Perspective,
        self_visits: f64,
    ) -> Array1<f64> {
        let num_actions = self
            .children
            .as_ref()
            .map(|c| c.len())
            .expect("Cannot find distribution of unexpanded node!");

        P::distribution(
            perspective,
            &self.prior_probabilities.slice(s![..num_actions]),
            &self.scores.slice(s![..num_actions]),
            &self.visits.slice(s![..num_actions]),
            self_visits,
        )
    }

    fn set_prior_probabilities(&mut self, prior_probabilities: Array1<f64>) {
        self.prior_probabilities = prior_probabilities;
    }
}

pub struct Arena {
    root_visits: f64,
    root_score: f64,
    root_id: NodeId,
    nodes: Vec<Node>,
    max_children: usize,
}

impl Arena {
    pub(crate) fn with_root(max_children: usize) -> Self {
        let mut this = Arena {
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
            index: self.nodes.len() - 1,
        }
    }

    fn create_root(&mut self) -> NodeId {
        self.nodes.truncate(0);
        self.root_id = self.create_node(None, 0);
        self.root_score = 0.;
        self.root_visits = 0.;
        self.root_id
    }

    pub(crate) fn expand_node(&mut self, node: NodeId, num_childs: usize) {
        debug_assert!(num_childs <= self.max_children);

        let children = (0..num_childs)
            .map(|i| self.create_node(Some(node), i))
            .collect();

        self.nodes[node.index].expand(children);
    }

    pub(crate) fn borrow(&self, node: NodeId) -> &Node {
        &self.nodes[node.index]
    }

    pub(crate) fn best_child<P: Policy, A: Action>(
        &self,
        perspective: Perspective,
        node: NodeId,
        actions: &[A],
    ) -> (NodeId, A) {
        self.nodes[node.index].best_child::<P, _>(perspective, self.visits(node), actions)
    }

    pub(crate) fn distribution<P: Policy>(
        &self,
        perspective: Perspective,
        node: NodeId,
    ) -> Array1<f64> {
        self.nodes[node.index].distribution::<P>(perspective, self.visits(node))
    }

    pub(crate) fn set_prior_probabilities(
        &mut self,
        node: NodeId,
        prior_probabilities: Array1<f64>,
    ) {
        self.nodes[node.index].set_prior_probabilities(prior_probabilities);
    }

    fn visits(&self, node: NodeId) -> f64 {
        match self.nodes[node.index].parent {
            None => {
                debug_assert!(node == self.root_id);
                self.root_visits
            }
            Some(parent) => {
                let index = self.nodes[node.index].index_in_parent;
                self.nodes[parent.index].visits[index]
            }
        }
    }

    pub(crate) fn visit(&mut self, node: NodeId) {
        match self.nodes[node.index].parent {
            None => {
                debug_assert!(node == self.root_id);
                self.root_visits += 1.;
            }
            Some(parent) => {
                let index = self.nodes[node.index].index_in_parent;
                self.nodes[parent.index].visits[index] += 1.;
            }
        }
    }

    pub(crate) fn add_score(&mut self, node: NodeId, score: f64) {
        match self.nodes[node.index].parent {
            None => {
                debug_assert!(node == self.root_id);
                self.root_score += score
            }
            Some(parent) => {
                let index = self.nodes[node.index].index_in_parent;
                self.nodes[parent.index].scores[index] += score;
            }
        }
    }
}
