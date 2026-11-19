//! Model metadata: variable shapes, constraints, counts, flattened names, and value ordering.
//!
//! Two orders matter and they differ for arrays of containers:
//!
//! * **Storage order** is how values sit in memory and in the unconstrained parameter vector:
//!   array elements in row-major order (the last array index varies fastest), each vector
//!   contiguous, each matrix in column-major order. It is the order of stanc3's deserializer.
//! * **Output order** is CmdStan's order for draws and flattened names: column-major over all
//!   indices (the first index varies fastest), with complex numbers as adjacent real and
//!   imaginary parts. [`VarInfo::output_names`] lists names in this order.
//!
//! Unconstrained names follow stanc3's `unconstrained_param_names`: column-major over the array
//! dimensions and the free dimensions, where a transform that changes the size (for example
//! `simplex[K]` to `K - 1` values) has a single free dimension.

use serde::{Deserialize, Serialize};

/// The program block a variable is declared in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Block {
    /// `data`.
    Data,
    /// `transformed data`.
    TransformedData,
    /// `parameters`.
    Parameters,
    /// `transformed parameters`.
    TransformedParameters,
    /// `generated quantities`.
    GeneratedQuantities,
}

/// The type of one element of a (possibly empty) array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BaseType {
    /// `int`.
    Int,
    /// `real`.
    Real,
    /// `complex`.
    Complex,
    /// `vector[n]`.
    Vector(usize),
    /// `row_vector[n]`.
    RowVector(usize),
    /// `matrix[rows, cols]`.
    Matrix(usize, usize),
    /// `complex_vector[n]`.
    ComplexVector(usize),
    /// `complex_row_vector[n]`.
    ComplexRowVector(usize),
    /// `complex_matrix[rows, cols]`.
    ComplexMatrix(usize, usize),
    /// `tuple(...)`: the component types are the [`VarInfo::slots`] of the variable.
    Tuple,
}

impl BaseType {
    /// The container dimensions (empty for scalars).
    pub fn dims(self) -> Vec<usize> {
        match self {
            BaseType::Int | BaseType::Real | BaseType::Complex | BaseType::Tuple => vec![],
            BaseType::Vector(n)
            | BaseType::RowVector(n)
            | BaseType::ComplexVector(n)
            | BaseType::ComplexRowVector(n) => vec![n],
            BaseType::Matrix(r, c) | BaseType::ComplexMatrix(r, c) => vec![r, c],
        }
    }

    /// True for complex element types (each element is two reals).
    pub fn is_complex(self) -> bool {
        matches!(
            self,
            BaseType::Complex
                | BaseType::ComplexVector(_)
                | BaseType::ComplexRowVector(_)
                | BaseType::ComplexMatrix(..)
        )
    }

    /// True for matrix types (stored column-major).
    pub fn is_matrix(self) -> bool {
        matches!(self, BaseType::Matrix(..) | BaseType::ComplexMatrix(..))
    }

    /// True for `int`.
    pub fn is_int(self) -> bool {
        self == BaseType::Int
    }

    /// Number of `f64` values in one element (0 for a tuple, whose size depends on its
    /// components; see [`VarInfo::num_constrained`]).
    pub fn num_values(self) -> usize {
        if self == BaseType::Tuple {
            return 0;
        }
        let n: usize = self.dims().iter().product();
        if self.is_complex() { 2 * n } else { n }
    }
}

/// A bound in a constraint: a single value known when the metadata is built, or a value per
/// element or one that depends on other variables (only known while the model runs).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Bound {
    /// The same bound for every element.
    Value(f64),
    /// A bound given by an expression (data dependent, elementwise, or parameter dependent).
    Expression,
}

/// The declared constraint of a variable, which for parameters selects the transform between the
/// constrained and unconstrained spaces.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// No constraint.
    None,
    /// `<lower=lb>`.
    Lower(Bound),
    /// `<upper=ub>`.
    Upper(Bound),
    /// `<lower=lb, upper=ub>`.
    LowerUpper(Bound, Bound),
    /// `<offset=mu, multiplier=sigma>`.
    OffsetMultiplier(Bound, Bound),
    /// `ordered[K]`.
    Ordered,
    /// `positive_ordered[K]`.
    PositiveOrdered,
    /// `simplex[K]`: `K - 1` unconstrained values.
    Simplex,
    /// `unit_vector[K]`: `K` unconstrained values.
    UnitVector,
    /// `sum_to_zero_vector[K]`: `K - 1` unconstrained values.
    SumToZero,
    /// `sum_to_zero_matrix[N, M]`: `(N - 1)(M - 1)` unconstrained values.
    SumToZeroMatrix,
    /// `cholesky_factor_corr[K]`: `K(K - 1)/2` unconstrained values.
    CholeskyFactorCorr,
    /// `cholesky_factor_cov[M, N]`: `N(N + 1)/2 + (M - N)N` unconstrained values.
    CholeskyFactorCov,
    /// `corr_matrix[K]`: `K(K - 1)/2` unconstrained values.
    CorrMatrix,
    /// `cov_matrix[K]`: `K + K(K - 1)/2` unconstrained values.
    CovMatrix,
    /// `column_stochastic_matrix[N, M]`: `(N - 1)M` unconstrained values.
    StochasticColumn,
    /// `row_stochastic_matrix[N, M]`: `N(M - 1)` unconstrained values.
    StochasticRow,
}

impl Constraint {
    /// The dimensions of the unconstrained values of one element of type `base`, as stanc3
    /// names them: a vector transform that drops a value keeps one dimension of the reduced
    /// size, the sum-to-zero and stochastic matrix transforms keep two reduced dimensions, and
    /// the Cholesky factor, correlation, and covariance transforms have a single free dimension.
    pub fn unconstrained_dims(self, base: BaseType) -> Vec<usize> {
        let d = base.dims();
        let at = |i: usize| d.get(i).copied().unwrap_or(0);
        let less = |i: usize| at(i).saturating_sub(1);
        match self {
            Constraint::Simplex | Constraint::SumToZero => vec![less(0)],
            Constraint::SumToZeroMatrix => vec![less(0), less(1)],
            Constraint::StochasticColumn => vec![less(0), at(1)],
            Constraint::StochasticRow => vec![at(0), less(1)],
            Constraint::CholeskyFactorCorr
            | Constraint::CholeskyFactorCov
            | Constraint::CorrMatrix
            | Constraint::CovMatrix => vec![unconstrained_size(base, self)],
            _ => d,
        }
    }
}

/// Number of unconstrained values of one element of type `base` under constraint `c`.
fn unconstrained_size(base: BaseType, c: Constraint) -> usize {
    let d = base.dims();
    let at = |i: usize| d.get(i).copied().unwrap_or(0);
    let tri = |k: usize| k * k.saturating_sub(1) / 2;
    match c {
        Constraint::Simplex | Constraint::SumToZero => at(0).saturating_sub(1),
        Constraint::SumToZeroMatrix => at(0).saturating_sub(1) * at(1).saturating_sub(1),
        Constraint::CholeskyFactorCorr | Constraint::CorrMatrix => tri(at(0)),
        Constraint::CovMatrix => at(0) + tri(at(0)),
        Constraint::CholeskyFactorCov => {
            let (m, n) = (at(0), at(1));
            n * (n + 1) / 2 + m.saturating_sub(n) * n
        }
        Constraint::StochasticColumn => at(0).saturating_sub(1) * at(1),
        Constraint::StochasticRow => at(0) * at(1).saturating_sub(1),
        _ => base.num_values(),
    }
}

/// Shape, type, and constraint of one model variable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VarInfo {
    /// Name in the Stan program.
    pub name: String,
    /// Block the variable is declared in.
    pub block: Block,
    /// Array dimensions, outermost first (empty if the variable is not an array).
    pub array_dims: Vec<usize>,
    /// Element type.
    pub base: BaseType,
    /// Declared constraint.
    pub constraint: Constraint,
    /// For a tuple (or an array of tuples): the component types in order. Their names are
    /// ignored; their array dimensions, types, and constraints apply to each tuple element.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slots: Vec<VarInfo>,
}

impl VarInfo {
    /// A variable.
    pub fn new(
        name: impl Into<String>,
        block: Block,
        array_dims: Vec<usize>,
        base: BaseType,
        constraint: Constraint,
    ) -> Self {
        VarInfo {
            name: name.into(),
            block,
            array_dims,
            base,
            constraint,
            slots: Vec::new(),
        }
    }

    /// A tuple variable (an array of tuples if `array_dims` is not empty) with components
    /// `slots`.
    pub fn tuple(
        name: impl Into<String>,
        block: Block,
        array_dims: Vec<usize>,
        slots: Vec<VarInfo>,
    ) -> Self {
        VarInfo {
            name: name.into(),
            block,
            array_dims,
            base: BaseType::Tuple,
            constraint: Constraint::None,
            slots,
        }
    }

    /// True for a tuple or an array of tuples.
    pub fn is_tuple(&self) -> bool {
        self.base == BaseType::Tuple
    }

    pub(crate) fn renamed(&self, name: String) -> VarInfo {
        VarInfo {
            name,
            ..self.clone()
        }
    }

    /// Number of constrained values of one array element.
    pub fn element_constrained(&self) -> usize {
        if self.is_tuple() {
            self.slots.iter().map(VarInfo::num_constrained).sum()
        } else {
            self.base.num_values()
        }
    }

    /// Number of unconstrained values of one array element.
    pub fn element_unconstrained(&self) -> usize {
        if self.is_tuple() {
            self.slots.iter().map(VarInfo::num_unconstrained).sum()
        } else {
            unconstrained_size(self.base, self.constraint)
        }
    }

    /// An unconstrained `real` parameter.
    pub fn real_param(name: impl Into<String>) -> Self {
        Self::new(
            name,
            Block::Parameters,
            vec![],
            BaseType::Real,
            Constraint::None,
        )
    }

    /// All dimensions: array dimensions, then container dimensions.
    pub fn dims(&self) -> Vec<usize> {
        let mut d = self.array_dims.clone();
        d.extend(self.base.dims());
        d
    }

    /// Number of array elements.
    pub fn num_elements(&self) -> usize {
        self.array_dims.iter().product()
    }

    /// Number of constrained `f64` values (complex numbers count twice).
    pub fn num_constrained(&self) -> usize {
        self.num_elements() * self.element_constrained()
    }

    /// Number of unconstrained values. Only meaningful for parameters.
    pub fn num_unconstrained(&self) -> usize {
        self.num_elements() * self.element_unconstrained()
    }

    /// Names of a tuple's components, element by element (array indices column-major), as
    /// stanc3 writes them: `p:1`, `q.2.1:2.3`.
    fn tuple_names(&self, names: impl Fn(&VarInfo) -> Vec<String>) -> Vec<String> {
        let mut out = Vec::new();
        for idx in column_major_indices(&self.array_dims) {
            let prefix = indexed_name(&self.name, &idx);
            for (k, slot) in self.slots.iter().enumerate() {
                out.extend(names(&slot.renamed(format!("{prefix}:{}", k + 1))));
            }
        }
        out
    }

    /// Flattened names in output order (`beta.1`, `Sigma.2.1`, `z.real`, `p:1`).
    pub fn output_names(&self) -> Vec<String> {
        if self.is_tuple() {
            return self.tuple_names(VarInfo::output_names);
        }
        let mut out = Vec::with_capacity(self.num_constrained());
        for idx in column_major_indices(&self.dims()) {
            let name = indexed_name(&self.name, &idx);
            if self.base.is_complex() {
                out.push(format!("{name}.real"));
                out.push(format!("{name}.imag"));
            } else {
                out.push(name);
            }
        }
        out
    }

    /// Flattened names of the unconstrained values, as stanc3 generates them.
    pub fn unconstrained_names(&self) -> Vec<String> {
        if self.is_tuple() {
            return self.tuple_names(VarInfo::unconstrained_names);
        }
        let mut dims = self.array_dims.clone();
        dims.extend(self.constraint.unconstrained_dims(self.base));
        let mut out = Vec::with_capacity(self.num_unconstrained());
        for idx in column_major_indices(&dims) {
            let name = indexed_name(&self.name, &idx);
            if self.base.is_complex() {
                out.push(format!("{name}.real"));
                out.push(format!("{name}.imag"));
            } else {
                out.push(name);
            }
        }
        out
    }

    /// Storage-order position of each output-order value: `output[i] = storage[perm[i]]`.
    ///
    /// A tuple variable is stored element by element with the array indices in column-major
    /// order (stanc3's order for arrays of tuples), each element holding its components in
    /// order, each in its own storage order.
    fn output_permutation(&self) -> Vec<usize> {
        if self.is_tuple() {
            let mut perm = Vec::with_capacity(self.num_constrained());
            let mut offset = 0;
            for _ in 0..self.num_elements() {
                for slot in &self.slots {
                    perm.extend(slot.output_permutation().into_iter().map(|p| p + offset));
                    offset += slot.num_constrained();
                }
            }
            return perm;
        }
        let unit = if self.base.is_complex() { 2 } else { 1 };
        let dims = self.dims();
        let k = dims.len();
        // Dimensions from slowest to fastest in storage: arrays row-major, then a matrix
        // column-major (its column index is slower than its row index).
        let mut order: Vec<usize> = (0..k).collect();
        if self.base.is_matrix() {
            order.swap(k - 2, k - 1);
        }
        let mut stride = vec![0; k];
        let mut s = unit;
        for &d in order.iter().rev() {
            stride[d] = s;
            s *= dims[d];
        }
        let mut perm = Vec::with_capacity(self.num_constrained());
        for idx in column_major_indices(&dims) {
            let p: usize = idx.iter().zip(&stride).map(|(i, st)| (i - 1) * st).sum();
            perm.extend(p..p + unit);
        }
        perm
    }

    /// Reorders values of this variable from storage order to output order.
    ///
    /// # Panics
    /// If `values` does not have [`VarInfo::num_constrained`] elements.
    pub fn to_output_order(&self, values: &[f64]) -> Vec<f64> {
        assert_eq!(
            values.len(),
            self.num_constrained(),
            "{}: wrong number of values",
            self.name
        );
        self.output_permutation()
            .into_iter()
            .map(|p| values[p])
            .collect()
    }

    /// Reorders values of this variable from output order to storage order.
    ///
    /// # Panics
    /// If `values` does not have [`VarInfo::num_constrained`] elements.
    pub fn to_storage_order(&self, values: &[f64]) -> Vec<f64> {
        assert_eq!(
            values.len(),
            self.num_constrained(),
            "{}: wrong number of values",
            self.name
        );
        let mut out = vec![0.0; values.len()];
        for (i, p) in self.output_permutation().into_iter().enumerate() {
            out[p] = values[i];
        }
        out
    }
}

fn indexed_name(name: &str, idx: &[usize]) -> String {
    let mut s = name.to_string();
    for i in idx {
        s.push('.');
        s.push_str(&i.to_string());
    }
    s
}

/// Every one-based index tuple of `dims` with the first index varying fastest. A scalar
/// (`dims` empty) has one empty tuple; any zero dimension gives no tuples.
pub fn column_major_indices(dims: &[usize]) -> Vec<Vec<usize>> {
    let n: usize = dims.iter().product();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let mut idx = vec![1; dims.len()];
    for _ in 0..n {
        out.push(idx.clone());
        for (k, d) in dims.iter().enumerate() {
            if idx[k] < *d {
                idx[k] += 1;
                break;
            }
            idx[k] = 1;
        }
    }
    out
}

/// Which output blocks to include.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outputs {
    /// Include transformed parameters.
    pub transformed_parameters: bool,
    /// Include generated quantities.
    pub generated_quantities: bool,
}

impl Outputs {
    /// Parameters only.
    pub const PARAMETERS: Outputs = Outputs {
        transformed_parameters: false,
        generated_quantities: false,
    };
    /// Parameters, transformed parameters, and generated quantities.
    pub const ALL: Outputs = Outputs {
        transformed_parameters: true,
        generated_quantities: true,
    };

    /// True if variables of `block` are included.
    pub fn includes(self, block: Block) -> bool {
        match block {
            Block::Parameters => true,
            Block::TransformedParameters => self.transformed_parameters,
            Block::GeneratedQuantities => self.generated_quantities,
            Block::Data | Block::TransformedData => false,
        }
    }
}

/// Metadata of an instantiated model (its sizes may depend on the data).
///
/// Built once when the model is constructed; evaluation never rebuilds it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name.
    pub name: String,
    /// Compiler that produced the model (for example `stanc-rs 0.1.0`), or `None` for a
    /// hand-written model.
    pub compiler: Option<String>,
    /// Version of this runtime the model was built against.
    pub runtime_version: String,
    /// Version of stanmath-rs the model was built against.
    pub math_version: String,
    /// Declared variables of every block, in declaration order.
    pub variables: Vec<VarInfo>,
}

impl ModelInfo {
    /// Metadata with the current runtime version and no compiler.
    pub fn new(name: impl Into<String>, variables: Vec<VarInfo>) -> Self {
        ModelInfo {
            name: name.into(),
            compiler: None,
            runtime_version: crate::RUNTIME_VERSION.to_string(),
            math_version: crate::MATH_VERSION.to_string(),
            variables,
        }
    }

    /// Variables of one block.
    pub fn block(&self, block: Block) -> impl Iterator<Item = &VarInfo> {
        self.variables.iter().filter(move |v| v.block == block)
    }

    /// The parameters.
    pub fn parameters(&self) -> impl Iterator<Item = &VarInfo> {
        self.block(Block::Parameters)
    }

    /// Variables written to the output for `outputs`, in output order.
    pub fn outputs(&self, outputs: Outputs) -> impl Iterator<Item = &VarInfo> {
        // Stan writes parameters, then transformed parameters, then generated quantities.
        [
            Block::Parameters,
            Block::TransformedParameters,
            Block::GeneratedQuantities,
        ]
        .into_iter()
        .filter(move |b| outputs.includes(*b))
        .flat_map(move |b| self.block(b))
    }

    /// The variable called `name`.
    pub fn variable(&self, name: &str) -> Option<&VarInfo> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// Number of unconstrained parameters.
    pub fn num_unconstrained(&self) -> usize {
        self.parameters().map(VarInfo::num_unconstrained).sum()
    }

    /// Number of output values for `outputs`.
    pub fn num_outputs(&self, outputs: Outputs) -> usize {
        self.outputs(outputs).map(VarInfo::num_constrained).sum()
    }

    /// Flattened output names for `outputs`.
    pub fn output_names(&self, outputs: Outputs) -> Vec<String> {
        self.outputs(outputs)
            .flat_map(VarInfo::output_names)
            .collect()
    }

    /// Flattened names of the unconstrained parameters.
    pub fn unconstrained_names(&self) -> Vec<String> {
        self.parameters()
            .flat_map(VarInfo::unconstrained_names)
            .collect()
    }

    /// Serializes the metadata to JSON.
    pub fn to_json(&self) -> String {
        // Serialization of these plain types cannot fail; non-finite bounds become null.
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Parses metadata from JSON.
    pub fn from_json(s: &str) -> crate::Result<Self> {
        serde_json::from_str(s)
            .map_err(|e| crate::ModelError::data(format!("invalid model metadata: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(name: &str, arr: Vec<usize>, base: BaseType, c: Constraint) -> VarInfo {
        VarInfo::new(name, Block::Parameters, arr, base, c)
    }

    /// Names checked against stanc 2.40's generated `constrained_param_names` and
    /// `unconstrained_param_names` for the same declarations.
    #[test]
    fn names_match_stanc() {
        let a = p("a", vec![2], BaseType::Vector(3), Constraint::None);
        assert_eq!(
            a.output_names(),
            ["a.1.1", "a.2.1", "a.1.2", "a.2.2", "a.1.3", "a.2.3"]
        );
        assert_eq!(a.unconstrained_names(), a.output_names());

        let s = p("s", vec![2], BaseType::Vector(3), Constraint::Simplex);
        assert_eq!(
            s.unconstrained_names(),
            ["s.1.1", "s.2.1", "s.1.2", "s.2.2"]
        );
        assert_eq!(s.num_unconstrained(), 4);
        assert_eq!(s.num_constrained(), 6);

        let cov = p("S", vec![], BaseType::Matrix(2, 2), Constraint::CovMatrix);
        assert_eq!(cov.unconstrained_names(), ["S.1", "S.2", "S.3"]);
        assert_eq!(cov.output_names(), ["S.1.1", "S.2.1", "S.1.2", "S.2.2"]);

        let l = p(
            "L",
            vec![],
            BaseType::Matrix(3, 3),
            Constraint::CholeskyFactorCorr,
        );
        assert_eq!(l.num_unconstrained(), 3);

        let m = p("m", vec![], BaseType::Matrix(2, 3), Constraint::None);
        assert_eq!(m.unconstrained_names()[..3], ["m.1.1", "m.2.1", "m.1.2"]);

        // sum_to_zero_matrix[3, 4], column_stochastic_matrix[3, 2], row_stochastic_matrix[2, 3]:
        // two reduced dimensions.
        let b = p(
            "b",
            vec![],
            BaseType::Matrix(3, 4),
            Constraint::SumToZeroMatrix,
        );
        assert_eq!(b.unconstrained_names()[..3], ["b.1.1", "b.2.1", "b.1.2"]);
        assert_eq!(b.unconstrained_names().len(), 6);
        let c = p(
            "c",
            vec![],
            BaseType::Matrix(3, 2),
            Constraint::StochasticColumn,
        );
        assert_eq!(
            c.unconstrained_names(),
            ["c.1.1", "c.2.1", "c.1.2", "c.2.2"]
        );
        let d = p(
            "d",
            vec![],
            BaseType::Matrix(2, 3),
            Constraint::StochasticRow,
        );
        assert_eq!(
            d.unconstrained_names(),
            ["d.1.1", "d.2.1", "d.1.2", "d.2.2"]
        );
        let st = p("a", vec![], BaseType::Vector(3), Constraint::SumToZero);
        assert_eq!(st.unconstrained_names(), ["a.1", "a.2"]);
        // cholesky_factor_cov[4, 3] and corr_matrix[3]: one free dimension.
        let e = p(
            "e",
            vec![],
            BaseType::Matrix(4, 3),
            Constraint::CholeskyFactorCov,
        );
        assert_eq!(e.unconstrained_names().len(), 9);
        assert_eq!(e.unconstrained_names()[8], "e.9");
        let f = p("f", vec![], BaseType::Matrix(3, 3), Constraint::CorrMatrix);
        assert_eq!(f.unconstrained_names(), ["f.1", "f.2", "f.3"]);
        // array[2] cov_matrix[2]: the free dimension varies slowest after the array index.
        let g = p("g", vec![2], BaseType::Matrix(2, 2), Constraint::CovMatrix);
        assert_eq!(g.unconstrained_names()[..3], ["g.1.1", "g.2.1", "g.1.2"]);
        // tuple(real, vector[2]) p; array[2, 3] tuple(real<lower=0>, simplex[3]) q;
        let tp = VarInfo::tuple(
            "p",
            Block::Parameters,
            vec![],
            vec![
                VarInfo::real_param(""),
                p("", vec![], BaseType::Vector(2), Constraint::None),
            ],
        );
        assert_eq!(tp.output_names(), ["p:1", "p:2.1", "p:2.2"]);
        let q = VarInfo::tuple(
            "q",
            Block::Parameters,
            vec![2, 3],
            vec![
                p(
                    "",
                    vec![],
                    BaseType::Real,
                    Constraint::Lower(Bound::Value(0.0)),
                ),
                p("", vec![], BaseType::Vector(3), Constraint::Simplex),
            ],
        );
        assert_eq!(
            q.output_names()[..6],
            [
                "q.1.1:1",
                "q.1.1:2.1",
                "q.1.1:2.2",
                "q.1.1:2.3",
                "q.2.1:1",
                "q.2.1:2.1"
            ]
        );
        assert_eq!(
            q.unconstrained_names()[..4],
            ["q.1.1:1", "q.1.1:2.1", "q.1.1:2.2", "q.2.1:1"]
        );
        assert_eq!((q.num_constrained(), q.num_unconstrained()), (24, 18));
        let zm = p(
            "zm",
            vec![],
            BaseType::ComplexMatrix(2, 2),
            Constraint::None,
        );
        assert_eq!(
            zm.unconstrained_names()[..3],
            ["zm.1.1.real", "zm.1.1.imag", "zm.2.1.real"]
        );
    }

    #[test]
    fn unconstrained_sizes() {
        let sz = |b, c| p("x", vec![], b, c).num_unconstrained();
        assert_eq!(
            sz(BaseType::Matrix(4, 3), Constraint::CholeskyFactorCov),
            6 + 3
        );
        assert_eq!(sz(BaseType::Matrix(3, 4), Constraint::StochasticColumn), 8);
        assert_eq!(sz(BaseType::Matrix(3, 4), Constraint::StochasticRow), 9);
        assert_eq!(sz(BaseType::Matrix(3, 4), Constraint::SumToZeroMatrix), 6);
        assert_eq!(sz(BaseType::Vector(5), Constraint::UnitVector), 5);
        assert_eq!(sz(BaseType::Vector(5), Constraint::SumToZero), 4);
        assert_eq!(sz(BaseType::Vector(0), Constraint::Simplex), 0);
        assert_eq!(sz(BaseType::Complex, Constraint::None), 2);
    }

    #[test]
    fn order_round_trip() {
        // array[2] vector[3]: storage a[1][1..3], a[2][1..3]; output column-major.
        let a = p("a", vec![2], BaseType::Vector(3), Constraint::None);
        let storage = [11.0, 12.0, 13.0, 21.0, 22.0, 23.0];
        let out = a.to_output_order(&storage);
        assert_eq!(out, [11.0, 21.0, 12.0, 22.0, 13.0, 23.0]);
        assert_eq!(a.to_storage_order(&out), storage);
        // A matrix is stored column-major already.
        let m = p("m", vec![], BaseType::Matrix(2, 3), Constraint::None);
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(m.to_output_order(&v), v);
        // Complex parts stay adjacent.
        let z = p("z", vec![2], BaseType::Complex, Constraint::None);
        assert_eq!(
            z.output_names(),
            ["z.1.real", "z.1.imag", "z.2.real", "z.2.imag"]
        );
        assert_eq!(
            z.to_output_order(&[1.0, 2.0, 3.0, 4.0]),
            [1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn zero_sizes_and_scalars() {
        let e = p("e", vec![0], BaseType::Real, Constraint::None);
        assert!(e.output_names().is_empty());
        assert_eq!(e.num_unconstrained(), 0);
        let r = VarInfo::real_param("mu");
        assert_eq!(r.output_names(), ["mu"]);
        assert_eq!(r.unconstrained_names(), ["mu"]);
    }

    #[test]
    fn metadata_json_round_trip() {
        let info = ModelInfo::new(
            "m",
            vec![
                VarInfo::new("y", Block::Data, vec![], BaseType::Real, Constraint::None),
                p(
                    "sigma",
                    vec![],
                    BaseType::Real,
                    Constraint::Lower(Bound::Value(0.0)),
                ),
            ],
        );
        let back = ModelInfo::from_json(&info.to_json()).unwrap();
        assert_eq!(back, info);
        assert_eq!(back.num_unconstrained(), 1);
    }
}
