extern crate ndarray;

use ndarray::Array1;

mod mcts;
mod tree;

trait Policy: Copy {
    fn evaluate<F>(
        self,
        prior_probabilities: &Array1<F>,
        scores: &Array1<F>,
        visits: &Array1<F>,
        parent_visits: F,
    ) -> Array1<F>;
}

enum Perspective {
    PreferPositive,
    PreferNegative,
}

impl Perspective {
    fn minmax_arg<F>(self, array: Array1<F>) -> Option<usize>
    where
        F: Copy,
        F: PartialOrd,
        F: ::std::ops::Neg<Output = F>,
    {
        self.maybe_negate(array)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Less))
            .map(|(index, _)| index)
    }

    fn maybe_negate<F>(self, array: Array1<F>) -> Array1<F>
    where
        F: ::std::ops::Neg<Output = F>,
        F: Copy,
    {
        match self {
            PreferPositive => array,
            PreferNegative => -array,
        }
    }
}

fn main() {
    println!("Hello, world!");
}
