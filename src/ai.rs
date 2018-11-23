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
    session: UnsafeCell<Session>,
}

impl WSession {
    unsafe fn get(&self) -> &mut Session {
        &mut *self.session.get()
    }
}

impl std::ops::Deref for WSession {
    type Target = Session;

    fn deref(&self) -> &Session {
        unsafe { &*self.session.get() }
    }
}

impl std::ops::DerefMut for WSession {
    fn deref_mut(&mut self) -> &mut Session {
        unsafe { &mut *self.session.get() }
    }
}

unsafe impl Send for WSession {}
unsafe impl Sync for WSession {}

pub struct TensorflowEvaluator<G, P> {
    graph: Graph,
    converter: G,
    session: WSession,
    phantom: PhantomData<(G, P)>,
    x: String,
    y: String,
    y_: String,
    train: String,
}

fn split_op(op: &str) -> (String, i32) {
    let mut split = op.split(":");
    let name = split.next().unwrap();
    let index = split.next().unwrap();

    (name.to_owned(), str::parse(index).unwrap())
}

impl<G: TensorflowConverter, P: Policy> TensorflowEvaluator<G, P> {
    pub fn new(
        converter: G,
        graph: Graph,
        session: Session,
        x: String,
        y: String,
        y_: String,
        train: String,
    ) -> Self {
        TensorflowEvaluator {
            x,
            y,
            y_,
            train,
            graph,
            converter,
            session: WSession {
                session: UnsafeCell::new(session),
            },
            phantom: PhantomData,
        }
    }

    pub fn load_model_from_file(
        game: G,
        filename: &str,
        x: String,
        y: String,
        y_: String,
        train: String,
    ) -> Result<Self, std::io::Error> {
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

        Ok(Self::new(game, graph, session, x, y, y_, train))
    }

    pub fn save(
        &mut self,
        dir: &str,
        filename: &str,
        variables: &[&str],
    ) -> Result<(), WriteNpzError> {
        std::fs::create_dir_all(dir)?;
        let mut npz = NpzWriter::new(File::create(Path::new(dir).join(filename))?);

        for var in variables {
            let (name, index) = split_op(var);

            let tensor = self
                .graph
                .operation_by_name_required(&name)
                .and_then(|tf_var| {
                    let mut args = SessionRunArgs::new();
                    let result = args.request_fetch(&tf_var, index);

                    self.session.run(&mut args)?;

                    Ok(args.fetch::<f64>(result)?)
                }).unwrap();

            npz.add_array(*var, &ArrayBase::from(&*tensor))?;
        }

        Ok(())
    }

    pub fn train(&mut self, replay_buffer: &Vec<(FullState<G>, Array1<f64>)>) {
        let (tx, ty) = self
            .converter
            .select(replay_buffer.into_iter().map(|(state, dist)| {
                (
                    self.converter.nn_state_representation(&state),
                    self.converter
                        .normalize_distribution(&state, dist.to_owned()),
                )
            }));

        let mut args = SessionRunArgs::new();

        let (xname, xi) = split_op(&self.x);

        self.graph
            .operation_by_name_required(&xname)
            .and_then(|x| {
                let (yname, yi) = split_op(&self.y);

                let y = self.graph.operation_by_name_required(&yname)?;

                args.add_feed(&x, xi, &tx);
                args.add_feed(&y, yi, &ty);

                let train =
                    args.request_fetch(&self.graph.operation_by_name_required(&self.train)?, 0);

                self.session.run(&mut args)?;

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

        let (xname, xi) = split_op(&self.x);
        let (yname, yi) = split_op(&self.y_);

        self.graph
            .operation_by_name_required(&xname)
            .and_then(|x| {
                args.add_feed(&x, xi, &tensor);

                let y = args.request_fetch(&self.graph.operation_by_name_required(&yname)?, yi);

                unsafe {
                    self.session.get().run(&mut args)?;
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
