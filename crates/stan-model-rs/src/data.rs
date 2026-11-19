//! Model data in CmdStan's JSON format, read and validated once when a model is constructed.
//!
//! Scalars are JSON numbers; a real may also be one of the strings `"NaN"`, `"Inf"`, `"-Inf"`,
//! `"Infinity"`, `"-Infinity"`. Vectors and row vectors are lists, matrices are lists of rows,
//! arrays are nested lists, and complex numbers are two-element lists `[re, im]`. An `int` must
//! be a JSON integer in Stan's 32-bit range: a real-valued JSON number is never converted to an
//! integer. A variable with no elements may be omitted.

use serde_json::{Map, Value};

use crate::meta::{BaseType, VarInfo};
use crate::transform::{Bounds, check};
use crate::{ModelError, Result};

/// A set of named values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Data {
    root: Map<String, Value>,
}

impl Data {
    /// No variables, for models without data.
    pub fn empty() -> Data {
        Data::default()
    }

    /// Parses a JSON object.
    pub fn from_json_str(s: &str) -> Result<Data> {
        let v: Value =
            serde_json::from_str(s).map_err(|e| ModelError::data(format!("invalid JSON: {e}")))?;
        Self::from_json_value(v)
    }

    /// Wraps an already parsed JSON object.
    pub fn from_json_value(v: Value) -> Result<Data> {
        match v {
            Value::Object(root) => Ok(Data { root }),
            _ => Err(ModelError::data("the JSON data must be an object")),
        }
    }

    /// Reads a JSON file.
    pub fn from_json_file(path: impl AsRef<std::path::Path>) -> Result<Data> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path)
            .map_err(|e| ModelError::data(format!("cannot read {}: {e}", path.display())))?;
        Self::from_json_str(&s)
    }

    /// True if variable `name` is present.
    pub fn contains(&self, name: &str) -> bool {
        self.root.contains_key(name)
    }

    /// Names of the variables present.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.root.keys().map(String::as_str)
    }

    /// Adds or replaces a variable.
    pub fn insert(&mut self, name: impl Into<String>, value: Value) {
        self.root.insert(name.into(), value);
    }

    /// Reads variable `var` with exactly the shape its metadata declares, returning its values in
    /// storage order (arrays row-major, matrices column-major, complex as `re, im`). Integers are
    /// returned as exact `f64` values.
    ///
    /// Fails with [`crate::ErrorKind::Data`] if the variable is missing (and not empty), has a
    /// different shape, holds a non-integer where an `int` is declared, or holds an integer
    /// outside Stan's 32-bit range.
    pub fn read(&self, var: &VarInfo) -> Result<Vec<f64>> {
        let name = &var.name;
        let mut dims = var.dims();
        if var.base.is_complex() {
            dims.push(2);
        }
        let value = match self.root.get(name) {
            Some(v) => v,
            None if dims.contains(&0) => return Ok(Vec::new()),
            None => {
                return Err(ModelError::data("variable not found in the data").with_variable(name));
            }
        };
        let mut row_major = Vec::with_capacity(dims.iter().product());
        flatten(value, &dims, var.base.is_int(), &mut row_major)
            .map_err(|e| e.with_variable(name.clone()))?;
        Ok(row_major_to_storage(var, row_major))
    }

    /// Reads `var` like [`Data::read`] and checks its declared constraint with the bounds in its
    /// metadata.
    pub fn read_checked(&self, var: &VarInfo) -> Result<Vec<f64>> {
        let bounds = Bounds::from_metadata(var).map_err(ModelError::into_data)?;
        self.read_with_bounds(var, &bounds)
    }

    /// Reads `var` like [`Data::read`] and checks its declared constraint with `bounds`.
    pub fn read_with_bounds(&self, var: &VarInfo, bounds: &Bounds) -> Result<Vec<f64>> {
        let x = self.read(var)?;
        check(var, bounds, &x).map_err(ModelError::into_data)?;
        Ok(x)
    }

    /// Reads an `int` scalar.
    pub fn int(&self, name: &str) -> Result<i64> {
        let var = VarInfo::new(
            name,
            crate::Block::Data,
            vec![],
            BaseType::Int,
            crate::Constraint::None,
        );
        Ok(self.read(&var)?[0] as i64)
    }

    /// Reads a `real` scalar.
    pub fn real(&self, name: &str) -> Result<f64> {
        let var = VarInfo::new(
            name,
            crate::Block::Data,
            vec![],
            BaseType::Real,
            crate::Constraint::None,
        );
        Ok(self.read(&var)?[0])
    }
}

/// Appends the values of `v`, which must have shape `dims`, in row-major order.
fn flatten(v: &Value, dims: &[usize], int: bool, out: &mut Vec<f64>) -> Result<()> {
    match dims.split_first() {
        None => {
            out.push(scalar(v, int)?);
            Ok(())
        }
        Some((&n, rest)) => {
            let items = match v {
                Value::Array(a) => a.as_slice(),
                _ => {
                    return Err(ModelError::data(format!(
                        "expected a list of {n} elements, found {}",
                        describe(v)
                    )));
                }
            };
            if items.len() != n {
                return Err(ModelError::data(format!(
                    "expected {n} elements, found {}",
                    items.len()
                )));
            }
            items.iter().try_for_each(|x| flatten(x, rest, int, out))
        }
    }
}

fn scalar(v: &Value, int: bool) -> Result<f64> {
    if int {
        let i = match v {
            Value::Number(n) => n.as_i64(),
            _ => None,
        };
        return match i {
            Some(i) if i32::try_from(i).is_ok() => Ok(i as f64),
            Some(i) => Err(ModelError::data(format!(
                "integer {i} is outside the 32-bit range of Stan's int"
            ))
            .with_value(i as f64)),
            None => Err(ModelError::data(format!(
                "expected an integer, found {}",
                describe(v)
            ))),
        };
    }
    match v {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| ModelError::data(format!("number {n} is not representable"))),
        Value::String(s) => match s.as_str() {
            "NaN" | "nan" => Ok(f64::NAN),
            "Inf" | "inf" | "Infinity" | "+Inf" | "+Infinity" => Ok(f64::INFINITY),
            "-Inf" | "-inf" | "-Infinity" => Ok(f64::NEG_INFINITY),
            _ => Err(ModelError::data(format!(
                "expected a number, found string {s:?}"
            ))),
        },
        _ => Err(ModelError::data(format!(
            "expected a number, found {}",
            describe(v)
        ))),
    }
}

fn describe(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => format!("boolean {b}"),
        Value::Number(n) => format!("number {n}"),
        Value::String(s) => format!("string {s:?}"),
        Value::Array(a) => format!("a list of {} elements", a.len()),
        Value::Object(_) => "an object".into(),
    }
}

/// Converts JSON's row-major order (matrices as lists of rows) to storage order (each matrix
/// column-major).
fn row_major_to_storage(var: &VarInfo, v: Vec<f64>) -> Vec<f64> {
    let (r, c) = match var.base {
        BaseType::Matrix(r, c) | BaseType::ComplexMatrix(r, c) => (r, c),
        _ => return v,
    };
    let unit = if var.base.is_complex() { 2 } else { 1 };
    let block = r * c * unit;
    if block == 0 {
        return v;
    }
    let mut out = Vec::with_capacity(v.len());
    for m in v.chunks(block) {
        for j in 0..c {
            for i in 0..r {
                let p = (i * c + j) * unit;
                out.extend_from_slice(&m[p..p + unit]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Block, Bound, Constraint, ErrorKind};

    fn d(base: BaseType, arr: Vec<usize>) -> VarInfo {
        VarInfo::new("v", Block::Data, arr, base, Constraint::None)
    }

    #[test]
    fn shapes_and_order() {
        let data =
            Data::from_json_str(r#"{"v": [[1, 2, 3], [4, 5, 6]], "z": [[1, 2], [3, 4]], "e": []}"#)
                .unwrap();
        // A matrix is a list of rows and is stored column-major.
        let m = data.read(&d(BaseType::Matrix(2, 3), vec![])).unwrap();
        assert_eq!(m, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // An array of vectors is stored row-major.
        let a = data.read(&d(BaseType::Vector(3), vec![2])).unwrap();
        assert_eq!(a, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let z = VarInfo::new(
            "z",
            Block::Data,
            vec![2],
            BaseType::Complex,
            Constraint::None,
        );
        assert_eq!(data.read(&z).unwrap(), [1.0, 2.0, 3.0, 4.0]);
        // Empty variables may be omitted or given as an empty list.
        let e = VarInfo::new(
            "missing",
            Block::Data,
            vec![0],
            BaseType::Real,
            Constraint::None,
        );
        assert!(data.read(&e).unwrap().is_empty());
    }

    #[test]
    fn errors() {
        let data = Data::from_json_str(r#"{"n": 2.5, "big": 3000000000, "v": [1, 2], "s": "Inf"}"#)
            .unwrap();
        let e = data.int("n").unwrap_err();
        assert_eq!(e.kind, ErrorKind::Data);
        assert_eq!(e.variable.as_deref(), Some("n"));
        assert_eq!(data.int("big").unwrap_err().kind, ErrorKind::Data);
        assert_eq!(data.real("missing").unwrap_err().kind, ErrorKind::Data);
        // No silent reshaping.
        assert!(data.read(&d(BaseType::Vector(3), vec![])).is_err());
        assert!(data.read(&d(BaseType::Real, vec![])).is_err());
        assert_eq!(data.real("s").unwrap(), f64::INFINITY);
        // A real is accepted for a real; an integer-valued JSON number is accepted too.
        assert_eq!(data.real("n").unwrap(), 2.5);
        assert!(Data::from_json_str("[1]").is_err());
        assert!(Data::from_json_str("{").is_err());
    }

    #[test]
    fn constraints_are_checked() {
        let data = Data::from_json_str(r#"{"N": 0, "y": [0.2, 0.8]}"#).unwrap();
        let n = VarInfo::new(
            "N",
            Block::Data,
            vec![],
            BaseType::Int,
            Constraint::Lower(Bound::Value(1.0)),
        );
        let e = data.read_checked(&n).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Data);
        assert_eq!(e.variable.as_deref(), Some("N"));
        let y = VarInfo::new(
            "y",
            Block::Data,
            vec![],
            BaseType::Vector(2),
            Constraint::Simplex,
        );
        assert_eq!(data.read_checked(&y).unwrap(), [0.2, 0.8]);
    }
}
