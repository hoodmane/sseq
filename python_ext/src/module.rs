//! Bindings for `algebra::module::FiniteDimensionalModule` (a.k.a. `FDModule`)
//! over the Steenrod algebra.
//!
//! The API is split into two types:
//!
//! - [`FiniteDimensionalModuleBuilder`] is the mutable builder. You allocate the graded
//!   pieces, name the basis elements, set/parse the action, and validate the
//!   Adem relations.
//! - [`FiniteDimensionalModule`] is the immutable result of `builder.build()`.
//!   It holds an `Arc<SteenrodModule>` and exposes read-only introspection
//!   plus `resolution()`. Because it is immutable and reference-counted, a
//!   `Resolution` built from it *shares* the same `Arc` rather than cloning
//!   the module data.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use algebra::{
    AdemAlgebra, MilnorAlgebra, SteenrodAlgebra,
    module::{FDModule as InnerFD, Module, SteenrodModule},
};
use anyhow::anyhow;
use bivec::BiVec;
use ext::chain_complex::FiniteChainComplex;
use fp::prime::{Prime, ValidPrime};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::resolution::Resolution;

/// A mutable builder for a finite-dimensional module over the (mod-`p`)
/// Steenrod algebra.
///
/// Construct one with a prime, a list of graded dimensions, and a choice of
/// algebra basis (`"adem"` or `"milnor"`). Then name the basis elements, set
/// the action (directly with [`set_action`] or via strings with
/// [`add_action`]), validate it, and call [`build`] to produce an immutable
/// [`FiniteDimensionalModule`].
#[pyclass(name = "FiniteDimensionalModuleBuilder", module = "sseq_ext")]
pub struct FiniteDimensionalModuleBuilder {
    /// `None` once the builder has been consumed by [`build`].
    inner: Option<InnerFD<SteenrodAlgebra>>,
    /// Cached `name -> (degree, index)` lookup, kept in sync with the module's
    /// basis element names so `add_action` can resolve generator names.
    name_to_gen: HashMap<String, (i32, usize)>,
}

impl FiniteDimensionalModuleBuilder {
    /// Borrow the inner module, or error if the builder was already consumed.
    fn module(&self) -> PyResult<&InnerFD<SteenrodAlgebra>> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("builder has already been consumed by build()"))
    }

    /// Mutably borrow the inner module, or error if already consumed.
    fn module_mut(&mut self) -> PyResult<&mut InnerFD<SteenrodAlgebra>> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("builder has already been consumed by build()"))
    }

    /// Rebuild the `name -> (degree, index)` map from the current module.
    fn rebuild_lookup(&mut self) {
        let mut map = HashMap::new();
        if let Some(inner) = &self.inner {
            let min = inner.min_degree();
            let max = inner.max_degree().unwrap_or(min - 1);
            for t in min..=max {
                for idx in 0..inner.dimension(t) {
                    map.insert(inner.basis_element_to_string(t, idx), (t, idx));
                }
            }
        }
        self.name_to_gen = map;
    }
}

#[pymethods]
impl FiniteDimensionalModuleBuilder {
    /// `FiniteDimensionalModuleBuilder(p, graded_dimension, algebra="adem", min_degree=0, name="")`.
    ///
    /// `graded_dimension[i]` is the dimension of the module in internal degree
    /// `min_degree + i`. Basis elements get default names `x{deg}_{idx}`;
    /// rename them with [`set_basis_element_name`].
    #[new]
    #[pyo3(signature = (p, graded_dimension, algebra="adem", min_degree=0, name=""))]
    fn new(
        p: u32,
        graded_dimension: Vec<usize>,
        algebra: &str,
        min_degree: i32,
        name: &str,
    ) -> PyResult<Self> {
        let p = ValidPrime::try_from(p)
            .map_err(|e| PyValueError::new_err(format!("Invalid prime: {e}")))?;
        let alg: SteenrodAlgebra = match algebra {
            "adem" => SteenrodAlgebra::AdemAlgebra(AdemAlgebra::new(p, false)),
            "milnor" => SteenrodAlgebra::MilnorAlgebra(MilnorAlgebra::new(p, false)),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid algebra {other:?}; expected \"adem\" or \"milnor\""
                )));
            }
        };
        let graded_dimension = BiVec::from_vec(min_degree, graded_dimension);
        let inner = InnerFD::new(Arc::new(alg), name.to_owned(), graded_dimension);
        let mut builder = Self {
            inner: Some(inner),
            name_to_gen: HashMap::new(),
        };
        builder.rebuild_lookup();
        Ok(builder)
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok(self.module()?.name.clone())
    }

    #[getter]
    fn prime(&self) -> PyResult<u32> {
        Ok(self.module()?.prime().as_u32())
    }

    fn min_degree(&self) -> PyResult<i32> {
        Ok(self.module()?.min_degree())
    }

    fn max_degree(&self) -> PyResult<Option<i32>> {
        Ok(self.module()?.max_degree())
    }

    /// Dimension of the module in internal degree `degree`.
    fn dimension(&self, degree: i32) -> PyResult<usize> {
        Ok(self.module()?.dimension(degree))
    }

    /// Name of basis element `idx` in degree `degree`.
    fn basis_element_to_string(&self, degree: i32, idx: usize) -> PyResult<String> {
        Ok(self.module()?.basis_element_to_string(degree, idx))
    }

    /// Rename basis element `idx` in degree `degree`.
    fn set_basis_element_name(&mut self, degree: i32, idx: usize, name: &str) -> PyResult<()> {
        let inner = self.module_mut()?;
        if degree < inner.min_degree() || idx >= inner.dimension(degree) {
            return Err(PyValueError::new_err(format!(
                "No basis element at (degree={degree}, idx={idx})"
            )));
        }
        inner.set_basis_element_name(degree, idx, name.to_owned());
        self.rebuild_lookup();
        Ok(())
    }

    /// Set the action of algebra basis element `(operation_degree,
    /// operation_idx)` on module basis element `(input_degree, input_idx)`.
    ///
    /// `output` is the coordinate vector (length `dimension(input_degree +
    /// operation_degree)`) of the result in the module's basis.
    fn set_action(
        &mut self,
        operation_degree: i32,
        operation_idx: usize,
        input_degree: i32,
        input_idx: usize,
        output: Vec<u32>,
    ) -> PyResult<()> {
        use algebra::Algebra;
        let inner = self.module_mut()?;
        let min_degree = inner.min_degree();
        let max_degree = inner.max_degree().unwrap_or(min_degree - 1);
        if operation_degree < 0 {
            return Err(PyValueError::new_err(format!(
                "operation_degree {operation_degree} must be non-negative"
            )));
        }
        if input_degree < min_degree || input_degree > max_degree {
            return Err(PyValueError::new_err(format!(
                "input_degree {input_degree} is outside the module's degree range [{min_degree}, {max_degree}]"
            )));
        }
        // `output_degree >= min_degree` follows from `input_degree >= min_degree`
        // and `operation_degree >= 0`. Guarding the upper bound keeps the
        // `algebra().dimension(operation_degree)` and `set_action` calls below
        // within the basis range the inner module computed at construction, so
        // they cannot panic across the FFI boundary.
        let output_degree = input_degree + operation_degree;
        if output_degree > max_degree {
            return Err(PyValueError::new_err(format!(
                "output_degree {output_degree} (= input_degree {input_degree} + operation_degree {operation_degree}) is outside the module's degree range [{min_degree}, {max_degree}]"
            )));
        }
        if operation_idx >= inner.algebra().dimension(operation_degree) {
            return Err(PyValueError::new_err(format!(
                "operation_idx {operation_idx} out of range for algebra degree {operation_degree}"
            )));
        }
        if input_idx >= inner.dimension(input_degree) {
            return Err(PyValueError::new_err(format!(
                "input_idx {input_idx} out of range for module degree {input_degree}"
            )));
        }
        let expected = inner.dimension(output_degree);
        if output.len() != expected {
            return Err(PyValueError::new_err(format!(
                "output has length {} but degree {output_degree} has dimension {expected}",
                output.len()
            )));
        }
        inner.set_action(
            operation_degree,
            operation_idx,
            input_degree,
            input_idx,
            &output,
        );
        Ok(())
    }

    /// Parse and add an action from a string like ``"Sq2 x0 = x2"`` or
    /// ``"Sq1 x0 = x1 + x3"``. Generator names must already be set.
    #[pyo3(signature = (action, overwrite=true))]
    fn add_action(&mut self, action: &str, overwrite: bool) -> PyResult<()> {
        // Snapshot the lookup so the closure does not borrow `self`.
        let lookup = self.name_to_gen.clone();
        let gen_to_idx = move |name: &str| -> anyhow::Result<(i32, usize)> {
            lookup
                .get(name)
                .copied()
                .ok_or_else(|| anyhow!("Unknown generator: {name}"))
        };
        let module = self.module_mut()?;
        let parsed = algebra::module::parse_action(module.algebra().as_ref(), gen_to_idx, action)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        module.apply_action(&parsed, overwrite);
        Ok(())
    }

    /// Extend the action from algebra generators to all algebra basis elements
    /// in degrees `(input_deg, output_deg)`, using the algebra's relations.
    fn extend_actions(&mut self, input_deg: i32, output_deg: i32) -> PyResult<()> {
        self.module_mut()?.extend_actions(input_deg, output_deg);
        Ok(())
    }

    /// Check that the action satisfies the algebra's relations between
    /// `input_deg` and `output_deg`. Raises `ValueError` if not.
    fn check_validity(&self, input_deg: i32, output_deg: i32) -> PyResult<()> {
        // The inner `check_validity` asserts `output_deg > input_deg`; guard
        // here so a bad range raises `ValueError` instead of aborting the
        // interpreter via a panic across the FFI boundary.
        if output_deg <= input_deg {
            return Err(PyValueError::new_err(format!(
                "output_deg {output_deg} must be greater than input_deg {input_deg}"
            )));
        }
        self.module()?
            .check_validity(input_deg, output_deg)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Convenience: extend the action over all decomposable operations and
    /// verify every relation across the whole module. Call this once after
    /// setting the action on algebra generators. Raises `ValueError` on an
    /// inconsistent module.
    fn validate(&mut self) -> PyResult<()> {
        let inner = self.module_mut()?;
        let min = inner.min_degree();
        let Some(max) = inner.max_degree() else {
            return Ok(());
        };
        for input_degree in (min..=max).rev() {
            for output_degree in (input_degree + 1)..=max {
                inner.extend_actions(input_degree, output_degree);
                inner
                    .check_validity(input_degree, output_degree)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Consume the builder and produce an immutable [`FiniteDimensionalModule`].
    ///
    /// This validates the module first (same as [`validate`]). After this the
    /// builder is spent and further use raises `ValueError`.
    fn build(&mut self) -> PyResult<FiniteDimensionalModule> {
        self.validate()?;
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("builder has already been consumed by build()"))?;
        self.name_to_gen.clear();
        Ok(FiniteDimensionalModule {
            module: Arc::new(Box::new(inner) as SteenrodModule),
        })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            None => "FiniteDimensionalModuleBuilder(<consumed>)".to_owned(),
            Some(inner) => {
                let name = if inner.name.is_empty() {
                    "<unnamed>".to_owned()
                } else {
                    format!("{:?}", inner.name)
                };
                let max_degree = inner
                    .max_degree()
                    .map_or_else(|| "None".to_owned(), |d| d.to_string());
                format!(
                    "FiniteDimensionalModuleBuilder(name={name}, p={}, min_degree={}, max_degree={max_degree})",
                    inner.prime().as_u32(),
                    inner.min_degree(),
                )
            }
        }
    }
}

/// An immutable finite-dimensional module over the (mod-`p`) Steenrod algebra,
/// produced by [`FiniteDimensionalModuleBuilder::build`].
///
/// Holds an `Arc<SteenrodModule>`; building a `Resolution` shares this `Arc`
/// rather than copying the module.
#[pyclass(name = "FiniteDimensionalModule", module = "sseq_ext")]
pub struct FiniteDimensionalModule {
    module: Arc<SteenrodModule>,
}

#[pymethods]
impl FiniteDimensionalModule {
    #[getter]
    fn name(&self) -> String {
        self.module.to_string()
    }

    #[getter]
    fn prime(&self) -> u32 {
        self.module.prime().as_u32()
    }

    fn min_degree(&self) -> i32 {
        self.module.min_degree()
    }

    fn max_degree(&self) -> Option<i32> {
        self.module.max_degree()
    }

    /// Dimension of the module in internal degree `degree`.
    fn dimension(&self, degree: i32) -> usize {
        self.module.dimension(degree)
    }

    /// Name of basis element `idx` in degree `degree`.
    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        self.module.basis_element_to_string(degree, idx)
    }

    /// Build a `Resolution` of this module, optionally backed by `save_dir`.
    ///
    /// The module is wrapped in a chain complex concentrated in homological
    /// degree 0 (`ccdz`) and resolved by a minimal free resolution. The
    /// underlying `Arc<SteenrodModule>` is shared, not cloned.
    #[pyo3(signature = (save_dir=None))]
    fn resolution(&self, save_dir: Option<PathBuf>) -> PyResult<Resolution> {
        let cc = FiniteChainComplex::ccdz(Arc::clone(&self.module));
        let res = ext::resolution::Resolution::new_with_save(Arc::new(cc), save_dir)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Resolution {
            inner: Arc::new(res),
        })
    }

    fn __repr__(&self) -> String {
        let name = self.module.to_string();
        let name = if name.is_empty() {
            "<unnamed>".to_owned()
        } else {
            format!("{name:?}")
        };
        let max_degree = self
            .module
            .max_degree()
            .map_or_else(|| "None".to_owned(), |d| d.to_string());
        format!(
            "FiniteDimensionalModule(name={name}, p={}, min_degree={}, max_degree={max_degree})",
            self.module.prime().as_u32(),
            self.module.min_degree(),
        )
    }
}
