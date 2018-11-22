use std::cell::UnsafeCell;
use std::fs::File;
use std::io::Read;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, ArrayBase, ArrayView1};
use ndarray_npy::{NpzWriter, WriteNpzError};
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor};

use super::{FullState, Game, Perspective};
use evaluators::StateEvaluator;
use policies::{Policy, SelectionStrategy, QU};

pub trait TensorflowConverter: Game {
    type D;

    fn nn_state_representation(&self, &FullState<Self>) -> Tensor<f64>;
    fn to_distribution(&self, &FullState<Self>, Tensor<f64>) -> Array1<f64>;
    fn normalize_distribution(&self, &FullState<Self>, Array1<f64>) -> Self::D;
    fn select(&self, impl Iterator<Item = (Tensor<f64>, Self::D)>) -> (Tensor<f64>, Tensor<f64>);
}

struct WSession {
    session: Session,
}

impl std::ops::Deref for WSession {
    type Target = Session;

    fn deref(&self) -> &Session {
        &self.session
    }
}

impl std::ops::DerefMut for WSession {
    fn deref_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}

unsafe impl Send for WSession {}

pub struct TensorflowEvaluator<G, P> {
    graph: Graph,
    converter: G,
    session: Mutex<UnsafeCell<WSession>>,
    phantom: PhantomData<(G, P)>,
}

impl<G: TensorflowConverter, P: Policy> TensorflowEvaluator<G, P> {
    pub fn new(converter: G, graph: Graph, session: Session) -> Self {
        TensorflowEvaluator {
            graph,
            converter,
            session: Mutex::new(UnsafeCell::new(WSession { session })),
            phantom: PhantomData,
        }
    }

    pub fn load_model_from_file(game: G, filename: &str) -> Result<Self, std::io::Error> {
        if !Path::new(filename).exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Could not find model at {}.", filename),
            ));
        }

        let mut proto = Vec::new();
        File::open(filename)?.read_to_end(&mut proto)?;

        let mut graph = Graph::new();
        match graph.import_graph_def(&proto, &ImportGraphDefOptions::new()) {
            Err(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Could not build tensorflow graph.",
                ))
            }
            Ok(_) => {}
        };

        let mut session = match Session::new(&SessionOptions::new(), &graph) {
            Err(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Could not create tensorflow session.",
                ))
            }
            Ok(session) => session,
        };

        graph
            .operation_by_name_required("init")
            .and_then(|init| {
                let mut args = SessionRunArgs::new();
                args.add_target(&init);

                session.run(&mut args)?;
                Ok(())
            }).unwrap();

        Ok(Self::new(game, graph, session))
    }

    pub fn save(&mut self, filename: &str, variables: &[&str]) -> Result<(), WriteNpzError> {
        let mut npz = NpzWriter::new(File::create(filename)?);

        for var in variables {
            let mut split = var.split(":");
            let name = split.next().unwrap();
            let index = str::parse(split.next().unwrap()).unwrap();

            let tensor = self
                .graph
                .operation_by_name_required(name)
                .and_then(|tf_var| {
                    let mut args = SessionRunArgs::new();
                    let result = args.request_fetch(&tf_var, index);

                    let session = self.session.get_mut().expect("Lock to not be poisoned.");

                    unsafe {
                        (*session.get()).run(&mut args)?;
                    }

                    Ok(args.fetch::<f64>(result)?)
                }).unwrap();

            npz.add_array(*var, &ArrayBase::from(&*tensor))?;
        }

        Ok(())
    }

    pub fn train(&mut self, replay_buffer: Vec<(FullState<G>, Array1<f64>)>) {
        let (tx, ty) = self
            .converter
            .select(replay_buffer.into_iter().map(|(state, dist)| {
                (
                    self.converter.nn_state_representation(&state),
                    self.converter.normalize_distribution(&state, dist),
                )
            }));

        let mut args = SessionRunArgs::new();

        self.graph
            .operation_by_name_required("x")
            .and_then(|x| {
                let y = self.graph.operation_by_name_required("y")?;

                args.add_feed(&x, 0, &tx);
                args.add_feed(&y, 0, &ty);

                let train = args.request_fetch(&self.graph.operation_by_name_required("Adam")?, 0);

                let session = self.session.get_mut().expect("Lock to not be poisoned.");
                unsafe {
                    (*session.get()).run(&mut args)?;
                }

                args.fetch::<i32>(train)?;

                Ok(())
            }).unwrap()
    }
}

impl<G: TensorflowConverter, P: Policy> StateEvaluator<G> for TensorflowEvaluator<G, P> {
    type TargetPolicy = P;

    fn evaluate(&self, state: &FullState<G>) -> Array1<f64> {
        let tensor = self.converter.nn_state_representation(state);

        let mut args = SessionRunArgs::new();

        // println!(
        //     "{:?}",
        //     self.graph
        //         .operation_iter()
        //         .map(|o| o.name())
        //         .collect::<Vec<_>>()
        // );
        // println!("");

        self.graph
            .operation_by_name_required("x")
            .and_then(|x| {
                args.add_feed(&x, 0, &tensor);

                let y =
                    args.request_fetch(&self.graph.operation_by_name_required("y_/Softmax")?, 0);

                let session = self.session.lock().expect("Lock to not be poisoned.");
                unsafe {
                    (*session.get()).run(&mut args)?;
                }

                Ok(self.converter.to_distribution(state, args.fetch(y)?))
            }).unwrap()
    }
}

#[derive(Copy, Clone)]
pub struct UCTAI<S: SelectionStrategy> {
    phantom: PhantomData<S>,
}

impl<S: SelectionStrategy> QU for UCTAI<S> {
    type Strategy = S;

    fn q(
        prior_probabilities: &ArrayView1<f64>,
        _scores: &ArrayView1<f64>,
        _visits: &ArrayView1<f64>,
        _parent_visits: f64,
    ) -> Array1<f64> {
        prior_probabilities.to_owned()
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
pub struct BestProbability<S: SelectionStrategy> {
    phantom: PhantomData<S>,
}

impl<S: SelectionStrategy> Policy for BestProbability<S> {
    type Strategy = S;

    fn distribution(
        _perspective: Perspective,
        prior_probabilities: &ArrayView1<f64>,
        _scores: &ArrayView1<f64>,
        _visits: &ArrayView1<f64>,
        _parent_visits: f64,
    ) -> Array1<f64> {
        prior_probabilities.to_owned()
    }
}
