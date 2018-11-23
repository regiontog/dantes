use std::f64;
use std::marker::PhantomData;

use ndarray::{Array1, ArrayView1};
use rand;
use rand::seq::SliceRandom;
use rand::Rng;

use super::Perspective;

pub trait SelectionStrategy {
    fn select(Perspective, &Array1<f64>) -> Option<usize>;
}

pub struct RandomS;

impl SelectionStrategy for RandomS {
    fn select(perspective: Perspective, distribution: &Array1<f64>) -> Option<usize> {
        let len = distribution.len();

        if len != 0 {
            Some(rand::thread_rng().gen_range(0, len))
        } else {
            None
        }
    }
}

pub struct MinMax;

impl SelectionStrategy for MinMax {
    fn select(perspective: Perspective, distribution: &Array1<f64>) -> Option<usize> {
        perspective
            .maybe_negate(distribution.to_owned())
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Less))
            .map(|(index, _)| index)
    }
}

pub struct WeightedRandom;

impl SelectionStrategy for WeightedRandom {
    fn select(perspective: Perspective, distribution: &Array1<f64>) -> Option<usize> {
        let mut distribution = perspective.maybe_negate(distribution.to_owned());

        let min = *distribution
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Greater))?;

        distribution = distribution.mapv(|v| v - min + 1.);

        let indices: Vec<_> = (0..distribution.len()).collect();

        indices
            .choose_weighted(&mut rand::thread_rng(), |&i| distribution[i])
            .ok()
            .map(|&i| i)
    }
}

pub trait Policy {
    type Strategy: SelectionStrategy;

    fn evaluate(
        perspective: Perspective,
        prior_probabilities: &ArrayView1<f64>,
        scores: &ArrayView1<f64>,
        visits: &ArrayView1<f64>,
        parent_visits: f64,
    ) -> Option<usize> {
        Self::Strategy::select(
            perspective,
            &Self::distribution(
                perspective,
                prior_probabilities,
                scores,
                visits,
                parent_visits,
            ),
        )
    }

    fn distribution(
        Perspective,
        &ArrayView1<f64>,
        &ArrayView1<f64>,
        &ArrayView1<f64>,
        f64,
    ) -> Array1<f64>;
}

pub trait QU {
    type Strategy: SelectionStrategy;

    fn q(&ArrayView1<f64>, &ArrayView1<f64>, &ArrayView1<f64>, f64) -> Array1<f64>;
    fn u(&ArrayView1<f64>, &ArrayView1<f64>, &ArrayView1<f64>, f64) -> Array1<f64>;
}

impl<T: QU> Policy for T {
    type Strategy = T::Strategy;

    fn distribution(
        perspective: Perspective,
        prior_probabilities: &ArrayView1<f64>,
        scores: &ArrayView1<f64>,
        visits: &ArrayView1<f64>,
        parent_visits: f64,
    ) -> Array1<f64> {
        Self::q(prior_probabilities, scores, visits, parent_visits)
            + perspective.maybe_negate(Self::u(prior_probabilities, scores, visits, parent_visits))
    }
}

#[derive(Copy, Clone)]
pub struct UCT<S: SelectionStrategy> {
    phantom: PhantomData<S>,
}

impl<S: SelectionStrategy> QU for UCT<S> {
    type Strategy = S;

    fn q(
        _prior_probabilities: &ArrayView1<f64>,
        scores: &ArrayView1<f64>,
        visits: &ArrayView1<f64>,
        _parent_visits: f64,
    ) -> Array1<f64> {
        scores.to_owned()
    }

    fn u(
        _prior_probabilities: &ArrayView1<f64>,
        _scores: &ArrayView1<f64>,
        visits: &ArrayView1<f64>,
        parent_visits: f64,
    ) -> Array1<f64> {
        3. * (parent_visits.ln() / visits.mapv(|v| v + 1.)).mapv(f64::sqrt)
    }
}

#[derive(Copy, Clone)]
pub struct Random {}

impl Policy for Random {
    type Strategy = MinMax;

    fn distribution(
        _perspective: Perspective,
        prior_probabilities: &ArrayView1<f64>,
        _scores: &ArrayView1<f64>,
        _visits: &ArrayView1<f64>,
        _parent_visits: f64,
    ) -> Array1<f64> {
        let len = prior_probabilities.len();
        let mut prior_probabilities = prior_probabilities.to_owned();

        if len != 0 {
            let rnd_index = rand::thread_rng().gen_range(0, len);
            prior_probabilities[rnd_index] = f64::INFINITY;
        }

        prior_probabilities
    }
}

#[derive(Copy, Clone)]
pub struct BestQuality<S: SelectionStrategy> {
    phantom: PhantomData<S>,
}

impl<S: SelectionStrategy> Policy for BestQuality<S> {
    type Strategy = S;

    fn distribution(
        _perspective: Perspective,
        _prior_probabilities: &ArrayView1<f64>,
        scores: &ArrayView1<f64>,
        visits: &ArrayView1<f64>,
        _parent_visits: f64,
    ) -> Array1<f64> {
        scores.to_owned()
    }
}
