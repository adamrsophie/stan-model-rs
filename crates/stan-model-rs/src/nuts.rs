//! An adapter for the NUTS sampler of the [`nuts_rs`] crate (feature `nuts`).
//!
//! The runtime does not implement or configure samplers. [`NutsModel`] only presents a
//! [`Model`] as the log density `nuts-rs` expects: the unconstrained log density with Jacobian and
//! without constants, its gradient, and the constrained parameters of each draw.

use std::collections::HashMap;

use nuts_rs::{CpuLogpFunc, CpuMathError, HasDims, LogpError};

use crate::{LogDensityOptions, Model, ModelError};

impl LogpError for ModelError {
    fn is_recoverable(&self) -> bool {
        self.kind.is_recoverable()
    }
}

/// A [`Model`] seen as a `nuts-rs` log density.
pub struct NutsModel<'a, M: ?Sized> {
    model: &'a M,
    options: LogDensityOptions,
}

impl<'a, M: Model + ?Sized> NutsModel<'a, M> {
    /// Samples with [`LogDensityOptions::SAMPLING`]: Jacobian included, constants dropped.
    pub fn new(model: &'a M) -> Self {
        NutsModel {
            model,
            options: LogDensityOptions::SAMPLING,
        }
    }
}

impl<M: Model + ?Sized> HasDims for NutsModel<'_, M> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([(
            "unconstrained_parameter".to_string(),
            self.model.num_unconstrained() as u64,
        )])
    }
}

impl<M: Model + ?Sized> CpuLogpFunc for NutsModel<'_, M> {
    type LogpError = ModelError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.model.num_unconstrained()
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, ModelError> {
        self.model
            .log_density_gradient_into(position, self.options, gradient)
    }

    /// The constrained parameters of a draw (generated quantities are the services layer's job,
    /// with its own random number stream).
    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        position: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        self.model
            .param_constrain(position)
            .map_err(|e| CpuMathError::ExpandError(e.to_string()))
    }
}
