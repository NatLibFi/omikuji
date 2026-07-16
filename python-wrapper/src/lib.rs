use itertools::Itertools;
use omikuji::rayon;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;
use std::sync::Mutex;

/// PyO3-compatible Model wrapper
#[pyclass]
struct Model {
    inner: omikuji::Model,
    /// Cached thread pool for parallel operations (mirrors Python implementation's self._thread_pool)
    thread_pool: Mutex<rayon::ThreadPool>,
    /// Cached process ID for fork detection (mirrors Python implementation's self._pid)
    cached_pid: usize,
}

#[pymethods]
impl Model {
    /// Load Omikuji model from the given directory.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let model = omikuji::Model::load(Path::new(&path)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load model: {}", e))
        })?;

        // Create default thread pool (rayon auto-detects thread count)
        let pool = rayon::ThreadPoolBuilder::new()
            .stack_size(32 * 1024 * 1024)
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create thread pool: {}",
                    e
                ))
            })?;

        Ok(Model {
            inner: model,
            thread_pool: Mutex::new(pool),
            cached_pid: std::process::id() as usize,
        })
    }

    /// Initialize/replace the thread pool for processing model predictions.
    ///
    /// If n_threads is set to 0, the number of threads is automatically chosen
    /// based on the number of available CPU cores.
    fn init_prediction_thread_pool(&mut self, n_threads: usize) -> PyResult<()> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .stack_size(32 * 1024 * 1024)
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create thread pool: {}",
                    e
                ))
            })?;

        *self.thread_pool.lock().unwrap() = pool;
        self.cached_pid = std::process::id() as usize;
        Ok(())
    }

    /// Save Omikuji model to the given directory.
    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(Path::new(&path)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to save model: {}", e))
        })?;
        Ok(())
    }

    /// Densify model weights to speed up prediction at the expense of memory usage.
    #[pyo3(signature = (max_sparse_density=0.1, n_threads=None))]
    fn densify_weights(
        &mut self,
        max_sparse_density: f32,
        n_threads: Option<usize>,
    ) -> PyResult<()> {
        // Check for fork: rebuild thread pool if PID changed (mirrors Python fork detection)
        let current_pid = std::process::id() as usize;
        if current_pid != self.cached_pid {
            let mut pool = self.thread_pool.lock().unwrap();
            *pool = rayon::ThreadPoolBuilder::new()
                .stack_size(32 * 1024 * 1024)
                .build()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create thread pool: {}",
                        e
                    ))
                })?;
            self.cached_pid = current_pid;
        }

        if let Some(n) = n_threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .stack_size(32 * 1024 * 1024)
                .build()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create thread pool: {}",
                        e
                    ))
                })?;
            pool.install(|| self.inner.densify_weights(max_sparse_density));
        } else {
            // Reuse the cached thread pool (mirrors Python implementation's self._thread_pool behavior)
            let guard = self.thread_pool.lock().unwrap();
            guard.install(|| self.inner.densify_weights(max_sparse_density));
        }
        Ok(())
    }

    /// Make predictions with Omikuji model.
    #[pyo3(signature = (feature_value_pairs, beam_size=None, top_k=None))]
    fn predict(
        &mut self,
        _py: Python,
        mut feature_value_pairs: Vec<(u32, f32)>,
        beam_size: Option<usize>,
        top_k: Option<usize>,
    ) -> PyResult<Vec<(u32, f32)>> {
        let beam_size = beam_size.unwrap_or(10);
        let top_k = top_k.unwrap_or(10);

        // Sort by feature index (mirrors Python implementation's behavior)
        feature_value_pairs.sort_by_key(|&(f, _)| f);

        // Validate indices are strictly ascending and in range (mirrors Python implementation's behavior)
        let n_features = self.n_features();
        if !feature_value_pairs.is_empty() {
            for ((f1, _), (f2, _)) in feature_value_pairs.iter().tuple_windows() {
                if !(*f1 < *f2) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Feature indices must be strictly ascending",
                    ));
                }
            }
            let (first, _) = &feature_value_pairs[0];
            let (last, _) = &feature_value_pairs[feature_value_pairs.len() - 1];
            if *first >= n_features as u32 || *last >= n_features as u32 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Feature index out of range",
                ));
            }
        }

        // Check for fork: rebuild thread pool if PID changed (mirrors Python fork detection)
        let current_pid = std::process::id() as usize;
        if current_pid != self.cached_pid {
            let mut pool = self.thread_pool.lock().unwrap();
            *pool = rayon::ThreadPoolBuilder::new()
                .stack_size(32 * 1024 * 1024)
                .build()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create thread pool: {}",
                        e
                    ))
                })?;
            self.cached_pid = current_pid;
        }

        // Use the cached thread pool for prediction (same as Python)
        let pool = self.thread_pool.lock().unwrap();
        let predictions = pool.install(|| self.inner.predict(&feature_value_pairs, beam_size));

        let result: Vec<(u32, f32)> = predictions
            .into_iter()
            .take(top_k)
            .map(|(label, score)| (label, score))
            .collect();
        Ok(result)
    }

    /// Get the expected dimension of feature vectors.
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    /// The number of trees in the forest model.
    #[getter]
    fn n_trees(&self) -> usize {
        self.inner.n_trees()
    }
}

/// Loss type enum for linear classifiers
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum LossType {
    Hinge,
    Log,
}

#[pymethods]
impl LossType {
    fn __repr__(&self) -> &'static str {
        match self {
            LossType::Hinge => "LossType.Hinge",
            LossType::Log => "LossType.Log",
        }
    }
}

impl From<omikuji::model::liblinear::LossType> for LossType {
    fn from(lt: omikuji::model::liblinear::LossType) -> Self {
        match lt {
            omikuji::model::liblinear::LossType::Hinge => LossType::Hinge,
            omikuji::model::liblinear::LossType::Log => LossType::Log,
        }
    }
}

impl From<LossType> for omikuji::model::liblinear::LossType {
    fn from(lt: LossType) -> Self {
        match lt {
            LossType::Hinge => omikuji::model::liblinear::LossType::Hinge,
            LossType::Log => omikuji::model::liblinear::LossType::Log,
        }
    }
}

/// Python-friendly HyperParam representation
#[pyclass(from_py_object)]
#[derive(Clone)]
struct HyperParam {
    #[pyo3(get, set)]
    n_trees: usize,
    #[pyo3(get, set)]
    min_branch_size: usize,
    #[pyo3(get, set)]
    max_depth: usize,
    #[pyo3(get, set)]
    centroid_threshold: f32,
    #[pyo3(get, set)]
    collapse_every_n_layers: usize,
    #[pyo3(get, set)]
    linear_loss_type: LossType,
    #[pyo3(get, set)]
    linear_eps: f32,
    #[pyo3(get, set)]
    linear_c: f32,
    #[pyo3(get, set)]
    linear_weight_threshold: f32,
    #[pyo3(get, set)]
    linear_max_iter: u32,
    #[pyo3(get, set)]
    cluster_k: usize,
    #[pyo3(get, set)]
    cluster_balanced: bool,
    #[pyo3(get, set)]
    cluster_eps: f32,
    #[pyo3(get, set)]
    cluster_min_size: usize,
    #[pyo3(get, set)]
    tree_structure_only: bool,
    #[pyo3(get, set)]
    train_trees_1_by_1: bool,
}

#[pymethods]
impl HyperParam {
    #[new]
    #[pyo3(signature = (
        n_trees=None,
        min_branch_size=None,
        max_depth=None,
        centroid_threshold=None,
        collapse_every_n_layers=None,
        linear_loss_type=None,
        linear_eps=None,
        linear_c=None,
        linear_weight_threshold=None,
        linear_max_iter=None,
        cluster_k=None,
        cluster_balanced=None,
        cluster_eps=None,
        cluster_min_size=None,
        tree_structure_only=false,
        train_trees_1_by_1=false,
        **kwargs
    ))]
    fn new(
        n_trees: Option<usize>,
        min_branch_size: Option<usize>,
        max_depth: Option<usize>,
        centroid_threshold: Option<f32>,
        collapse_every_n_layers: Option<usize>,
        linear_loss_type: Option<LossType>,
        linear_eps: Option<f32>,
        linear_c: Option<f32>,
        linear_weight_threshold: Option<f32>,
        linear_max_iter: Option<u32>,
        cluster_k: Option<usize>,
        cluster_balanced: Option<bool>,
        cluster_eps: Option<f32>,
        cluster_min_size: Option<usize>,
        tree_structure_only: bool,
        train_trees_1_by_1: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let default = omikuji::model::train::HyperParam::default();

        let mut hyper_param = Self {
            n_trees: n_trees.unwrap_or(default.n_trees),
            min_branch_size: min_branch_size.unwrap_or(default.min_branch_size),
            max_depth: max_depth.unwrap_or(default.max_depth),
            centroid_threshold: centroid_threshold.unwrap_or(default.centroid_threshold),
            collapse_every_n_layers: collapse_every_n_layers
                .unwrap_or(default.collapse_every_n_layers),
            linear_loss_type: linear_loss_type.unwrap_or(default.linear.loss_type.into()),
            linear_eps: linear_eps.unwrap_or(default.linear.eps),
            linear_c: linear_c.unwrap_or(default.linear.c),
            linear_weight_threshold: linear_weight_threshold
                .unwrap_or(default.linear.weight_threshold),
            linear_max_iter: linear_max_iter.unwrap_or(default.linear.max_iter),
            cluster_k: cluster_k.unwrap_or(default.cluster.k),
            cluster_balanced: cluster_balanced.unwrap_or(default.cluster.balanced),
            cluster_eps: cluster_eps.unwrap_or(default.cluster.eps),
            cluster_min_size: cluster_min_size.unwrap_or(default.cluster.min_size),
            tree_structure_only,
            train_trees_1_by_1,
        };

        // Allow dict-style passing for linear and cluster sub-params
        if let Some(kwargs) = kwargs {
            if let Some(linear) = kwargs.get_item("linear").ok().flatten() {
                if let Ok(linear_dict) = linear.cast::<PyDict>() {
                    hyper_param.linear_eps = linear_dict
                        .get_item("eps")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<f32>().ok())
                        .unwrap_or(hyper_param.linear_eps);
                    hyper_param.linear_c = linear_dict
                        .get_item("c")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<f32>().ok())
                        .unwrap_or(hyper_param.linear_c);
                    hyper_param.linear_weight_threshold = linear_dict
                        .get_item("weight_threshold")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<f32>().ok())
                        .unwrap_or(hyper_param.linear_weight_threshold);
                    hyper_param.linear_max_iter = linear_dict
                        .get_item("max_iter")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<u32>().ok())
                        .unwrap_or(hyper_param.linear_max_iter);
                    if let Some(loss_type) = linear_dict
                        .get_item("loss_type")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<LossType>().ok())
                    {
                        hyper_param.linear_loss_type = loss_type;
                    }
                }
            }
            if let Some(cluster) = kwargs.get_item("cluster").ok().flatten() {
                if let Ok(cluster_dict) = cluster.cast::<PyDict>() {
                    hyper_param.cluster_k = cluster_dict
                        .get_item("k")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<usize>().ok())
                        .unwrap_or(hyper_param.cluster_k);
                    hyper_param.cluster_balanced = cluster_dict
                        .get_item("balanced")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<bool>().ok())
                        .unwrap_or(hyper_param.cluster_balanced);
                    hyper_param.cluster_eps = cluster_dict
                        .get_item("eps")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<f32>().ok())
                        .unwrap_or(hyper_param.cluster_eps);
                    hyper_param.cluster_min_size = cluster_dict
                        .get_item("min_size")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<usize>().ok())
                        .unwrap_or(hyper_param.cluster_min_size);
                }
            }
        }

        Ok(hyper_param)
    }

    fn __repr__(&self) -> String {
        format!(
            "HyperParam(n_trees={}, min_branch_size={}, max_depth={}, centroid_threshold={:.4})",
            self.n_trees, self.min_branch_size, self.max_depth, self.centroid_threshold
        )
    }
}

/// Convert HyperParam to native type (internal use only)
fn hyper_param_to_native(hp: &HyperParam) -> omikuji::model::train::HyperParam {
    omikuji::model::train::HyperParam {
        n_trees: hp.n_trees,
        min_branch_size: hp.min_branch_size,
        max_depth: hp.max_depth,
        centroid_threshold: hp.centroid_threshold,
        collapse_every_n_layers: hp.collapse_every_n_layers,
        linear: omikuji::model::liblinear::HyperParam {
            loss_type: hp.linear_loss_type.into(),
            eps: hp.linear_eps,
            c: hp.linear_c,
            weight_threshold: hp.linear_weight_threshold,
            max_iter: hp.linear_max_iter,
        },
        cluster: omikuji::model::cluster::HyperParam {
            k: hp.cluster_k,
            balanced: hp.cluster_balanced,
            eps: hp.cluster_eps,
            min_size: hp.cluster_min_size,
        },
        tree_structure_only: hp.tree_structure_only,
        train_trees_1_by_1: hp.train_trees_1_by_1,
    }
}

/// Get the default training hyper-parameters.
#[pyfunction]
fn default_hyper_param() -> HyperParam {
    let default = omikuji::model::train::HyperParam::default();
    HyperParam {
        n_trees: default.n_trees,
        min_branch_size: default.min_branch_size,
        max_depth: default.max_depth,
        centroid_threshold: default.centroid_threshold,
        collapse_every_n_layers: default.collapse_every_n_layers,
        linear_loss_type: default.linear.loss_type.into(),
        linear_eps: default.linear.eps,
        linear_c: default.linear.c,
        linear_weight_threshold: default.linear.weight_threshold,
        linear_max_iter: default.linear.max_iter,
        cluster_k: default.cluster.k,
        cluster_balanced: default.cluster.balanced,
        cluster_eps: default.cluster.eps,
        cluster_min_size: default.cluster.min_size,
        tree_structure_only: default.tree_structure_only,
        train_trees_1_by_1: default.train_trees_1_by_1,
    }
}

/// Helper: create a default thread pool for a new Model
fn make_default_pool() -> PyResult<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .stack_size(32 * 1024 * 1024)
        .build()
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create thread pool: {}",
                e
            ))
        })
}

/// Helper: construct a Model with a thread pool
fn make_model(inner: omikuji::Model) -> PyResult<Model> {
    let pool = make_default_pool()?;
    Ok(Model {
        inner,
        thread_pool: Mutex::new(pool),
        cached_pid: std::process::id() as usize,
    })
}

/// Train a model with the given data file path and hyper-parameters.
#[pyfunction]
#[pyo3(signature = (data_path, hyper_param=None, n_threads=None))]
fn train_on_data(
    _py: Python,
    data_path: String,
    hyper_param: Option<&HyperParam>,
    n_threads: Option<usize>,
) -> PyResult<Model> {
    let hyper_param = match hyper_param {
        Some(hp) => hyper_param_to_native(hp),
        None => omikuji::model::train::HyperParam::default(),
    };

    if let Some(n) = n_threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .stack_size(32 * 1024 * 1024)
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create thread pool: {}",
                    e
                ))
            })?;
        let dataset = pool.install(|| {
            omikuji::DataSet::load_xc_repo_data_file(Path::new(&data_path)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load data: {}", e))
            })
        })?;
        let model = pool.install(|| hyper_param.train(dataset));
        make_model(model)
    } else {
        let dataset =
            omikuji::DataSet::load_xc_repo_data_file(Path::new(&data_path)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load data: {}", e))
            })?;
        let model = hyper_param.train(dataset);
        make_model(model)
    }
}

/// Initialize a simple logger that writes to stdout.
#[pyfunction]
fn init_logger() -> PyResult<()> {
    simple_logger::init().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to initialize logger: {}", e))
    })?;
    Ok(())
}

/// Python module definition
#[pymodule]
fn _omikuji(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<LossType>()?;
    m.add_class::<HyperParam>()?;
    m.add_function(wrap_pyfunction!(default_hyper_param, m)?)?;
    m.add_function(wrap_pyfunction!(train_on_data, m)?)?;
    m.add_function(wrap_pyfunction!(init_logger, m)?)?;
    Ok(())
}
