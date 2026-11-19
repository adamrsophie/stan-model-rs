//! Structured runtime errors.

use std::fmt;

/// The category of a [`ModelError`].
///
/// Inference code branches on the kind: a [`ErrorKind::Domain`] or [`ErrorKind::Constraint`]
/// error during a log density evaluation means "reject this proposal", while
/// [`ErrorKind::Data`], [`ErrorKind::Parameters`], [`ErrorKind::Fatal`], and
/// [`ErrorKind::Internal`] mean "stop".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ErrorKind {
    /// Input data is missing, malformed, has the wrong shape, or violates its declared
    /// constraints. Raised while constructing a model, before any inference.
    Data,
    /// The unconstrained parameter vector, or a vector of constrained values, has the wrong
    /// length or is not finite where a finite value is required.
    Parameters,
    /// A declared constraint of a parameter, transformed parameter, or generated quantity is
    /// violated (for example a constrained initial value outside its support).
    Constraint,
    /// A function argument is outside its domain while evaluating the model (Stan rejects the
    /// current parameter values).
    Domain,
    /// An index was out of range.
    Index,
    /// Containers had inconsistent sizes.
    Size,
    /// The program called `reject`.
    Reject,
    /// The program called `fatal_error`; inference must stop.
    Fatal,
    /// The operation is not supported by this model or by the runtime.
    Unsupported,
    /// A bug in the runtime or in a generated model (an invariant did not hold).
    Internal,
}

impl ErrorKind {
    /// True if inference may continue after this error by rejecting the current proposal.
    ///
    /// This matches Stan, where `std::domain_error` (argument and constraint violations, and
    /// `reject`) rejects the proposal and every other exception stops the algorithm.
    pub fn is_recoverable(self) -> bool {
        matches!(
            self,
            ErrorKind::Domain | ErrorKind::Constraint | ErrorKind::Reject
        )
    }
}

/// An error raised while constructing or evaluating a model.
///
/// The optional fields carry the context Stan prints in its messages: the model variable, the
/// mathematical function, and the offending value.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelError {
    /// Classification.
    pub kind: ErrorKind,
    /// Human readable explanation.
    pub message: String,
    /// Model variable involved (data variable, parameter, or output), if known.
    pub variable: Option<String>,
    /// Mathematical function that rejected its arguments, if known.
    pub function: Option<String>,
    /// Offending numerical value, if there is a single one.
    pub value: Option<f64>,
    /// The Stan source locations of the statements that were running, innermost first: the
    /// statement that failed, then the call sites of the user-defined functions it was in.
    /// Each is written as stanc3 writes them, for example
    /// `'model.stan', line 5, column 2 to column 20`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<String>,
}

impl ModelError {
    /// Creates an error with no context fields.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        ModelError {
            kind,
            message: message.into(),
            variable: None,
            function: None,
            value: None,
            locations: Vec::new(),
        }
    }

    /// A [`ErrorKind::Data`] error.
    pub fn data(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Data, message)
    }

    /// A [`ErrorKind::Domain`] error.
    pub fn domain(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Domain, message)
    }

    /// A [`ErrorKind::Parameters`] error.
    pub fn parameters(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Parameters, message)
    }

    /// A [`ErrorKind::Constraint`] error.
    pub fn constraint(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Constraint, message)
    }

    /// A [`ErrorKind::Internal`] error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }

    /// Sets the variable name (keeps an existing one, which is more specific).
    pub fn with_variable(mut self, name: impl Into<String>) -> Self {
        if self.variable.is_none() {
            self.variable = Some(name.into());
        }
        self
    }

    /// Sets the function name.
    pub fn with_function(mut self, name: impl Into<String>) -> Self {
        self.function = Some(name.into());
        self
    }

    /// Sets the offending value.
    pub fn with_value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    /// Returns the error with `context` prepended to its message.
    pub fn context(mut self, context: &str) -> Self {
        self.message = format!("{context}: {}", self.message);
        self
    }

    /// Adds the source location of the statement that was running when the error passed
    /// through a generated function.
    pub fn at(mut self, location: &str) -> Self {
        self.locations.push(location.to_string());
        self
    }

    /// Reclassifies the error as a [`ErrorKind::Data`] error. Generated code applies this to
    /// constraint checks in the data and transformed data blocks, whose failures are data
    /// problems rather than rejected proposals.
    pub fn into_data(mut self) -> Self {
        self.kind = ErrorKind::Data;
        self
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(v) = &self.variable {
            write!(f, "{v}: ")?;
        }
        write!(f, "{}", self.message)?;
        // As Stan's `rethrow_located` prints them.
        for loc in &self.locations {
            write!(f, " (in {loc})")?;
        }
        Ok(())
    }
}

/// Adds `table[statement]` to an error leaving a generated function. `statement` is the
/// position in the model's location table of the last statement that started, or any
/// out-of-range value (generated code uses `usize::MAX`) before the first one.
pub fn locate(e: ModelError, statement: usize, table: &[&str]) -> ModelError {
    match table.get(statement) {
        Some(loc) => e.at(loc),
        None => e,
    }
}

impl std::error::Error for ModelError {}

impl From<stanmath_rs::Error> for ModelError {
    fn from(e: stanmath_rs::Error) -> Self {
        use stanmath_rs::Error as E;
        let message = e.to_string();
        match e {
            E::InvalidParameter {
                function, value, ..
            }
            | E::InvalidObservation {
                function, value, ..
            } => ModelError::domain(message)
                .with_function(function)
                .with_value(value),
            E::DimensionMismatch { function, .. } => {
                ModelError::new(ErrorKind::Size, message).with_function(function)
            }
            E::NotPositiveDefinite { function }
            | E::NotSymmetric { function, .. }
            | E::NoConvergence { function, .. } => {
                ModelError::domain(message).with_function(function)
            }
            _ => ModelError::domain(message),
        }
    }
}

/// Result alias for model code.
pub type Result<T, E = ModelError> = std::result::Result<T, E>;

/// Raises a `reject` error with the printed message.
pub fn reject<T>(message: impl Into<String>) -> Result<T> {
    Err(ModelError::new(ErrorKind::Reject, message))
}

/// Raises a `fatal_error` with the printed message.
pub fn fatal_error<T>(message: impl Into<String>) -> Result<T> {
    Err(ModelError::new(ErrorKind::Fatal, message))
}
