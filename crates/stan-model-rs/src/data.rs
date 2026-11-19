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
        let s = quote_non_finite(s);
        let v: Value =
            serde_json::from_str(&s).map_err(|e| ModelError::data(format!("invalid JSON: {e}")))?;
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

    /// The data as a JSON string (CmdStan's format).
    pub fn to_json_string(&self) -> String {
        Value::Object(self.root.clone()).to_string()
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
        read_value(var, self.root.get(&var.name))
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

/// Reads the value `value` of `var` (or `None` if it is absent) into storage order.
fn read_value(var: &VarInfo, value: Option<&Value>) -> Result<Vec<f64>> {
    let name = &var.name;
    let mut dims = var.dims();
    if var.base.is_complex() {
        dims.push(2);
    }
    let value = match value {
        Some(v) => v,
        None if var.num_elements() == 0 || dims.contains(&0) => return Ok(Vec::new()),
        None => {
            return Err(ModelError::data("variable not found in the data").with_variable(name));
        }
    };
    if var.is_tuple() {
        return read_tuple(var, value);
    }
    let mut row_major = Vec::with_capacity(dims.iter().product());
    flatten(value, &dims, var.base.is_int(), &mut row_major)
        .map_err(|e| e.with_variable(name.clone()))?;
    Ok(row_major_to_storage(var, row_major))
}

/// Reads a tuple or an array of tuples. Each tuple is a JSON object with keys `"1"`, `"2"`, ...;
/// the elements are stored with the array indices in column-major order.
fn read_tuple(var: &VarInfo, value: &Value) -> Result<Vec<f64>> {
    let mut elements = Vec::with_capacity(var.num_elements());
    collect_elements(value, &var.array_dims, &mut elements)
        .map_err(|e| e.with_variable(var.name.clone()))?;
    // JSON nests arrays row-major; reorder to column-major.
    let dims = &var.array_dims;
    let mut out = Vec::with_capacity(var.num_constrained());
    for idx in crate::meta::column_major_indices(dims) {
        let mut p = 0;
        for (i, d) in idx.iter().zip(dims) {
            p = p * d + (i - 1);
        }
        let Value::Object(fields) = elements[p] else {
            return Err(ModelError::data(format!(
                "expected a tuple (an object with keys \"1\", \"2\", ...), found {}",
                describe(elements[p])
            ))
            .with_variable(var.name.clone()));
        };
        for (k, slot) in var.slots.iter().enumerate() {
            let slot = slot.renamed(format!("{}:{}", var.name, k + 1));
            out.extend(read_value(&slot, fields.get(&(k + 1).to_string()))?);
        }
    }
    Ok(out)
}

fn collect_elements<'a>(v: &'a Value, dims: &[usize], out: &mut Vec<&'a Value>) -> Result<()> {
    match dims.split_first() {
        None => {
            out.push(v);
            Ok(())
        }
        Some((&n, rest)) => match v {
            Value::Array(a) if a.len() == n => {
                a.iter().try_for_each(|x| collect_elements(x, rest, out))
            }
            _ => Err(ModelError::data(format!(
                "expected a list of {n} elements, found {}",
                describe(v)
            ))),
        },
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

/// Stan's JSON reader also accepts the bare tokens `NaN`, `Inf`, `-Inf`, `Infinity`, and
/// `-Infinity`, which strict JSON does not. They are quoted here (outside string literals), and
/// the quoted forms are then read like the string spellings Stan accepts.
fn quote_non_finite(s: &str) -> std::borrow::Cow<'_, str> {
    const TOKENS: [&str; 5] = ["-Infinity", "Infinity", "-Inf", "Inf", "NaN"];
    if !TOKENS.iter().any(|t| s.contains(t)) {
        return s.into();
    }
    let mut out = String::with_capacity(s.len() + 16);
    let (mut in_string, mut escaped) = (false, false);
    let mut rest = s;
    while let Some(c) = rest.chars().next() {
        if in_string {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
        } else if let Some(t) = TOKENS.iter().find(|t| rest.starts_with(**t)) {
            let after = rest[t.len()..].chars().next();
            if !after.is_some_and(|a| a.is_alphanumeric() || a == '_') {
                out.push('"');
                out.push_str(t);
                out.push('"');
                rest = &rest[t.len()..];
                continue;
            }
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out.into()
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
    fn tuples() {
        let data = Data::from_json_str(
            r#"{"q": [[{"1": 1, "2": [0.5, 0.5]}, {"1": 2, "2": [0.1, 0.9]}],
                      [{"1": 3, "2": [0.2, 0.8]}, {"1": 4, "2": [0.3, 0.7]}]]}"#,
        )
        .unwrap();
        let slots = vec![
            VarInfo::new(
                "",
                Block::Data,
                vec![],
                BaseType::Int,
                Constraint::Lower(Bound::Value(0.0)),
            ),
            VarInfo::new(
                "",
                Block::Data,
                vec![],
                BaseType::Vector(2),
                Constraint::Simplex,
            ),
        ];
        let q = VarInfo::tuple("q", Block::Data, vec![2, 2], slots);
        // Elements in column-major order: q[1,1], q[2,1], q[1,2], q[2,2].
        assert_eq!(
            data.read_checked(&q).unwrap(),
            [1.0, 0.5, 0.5, 3.0, 0.2, 0.8, 2.0, 0.1, 0.9, 4.0, 0.3, 0.7]
        );
        let bad = Data::from_json_str(r#"{"q": [[{"1": 1}]]}"#).unwrap();
        assert_eq!(bad.read(&q).unwrap_err().kind, ErrorKind::Data);
        let neg = Data::from_json_str(
            r#"{"q": [[{"1": -1, "2": [0.5, 0.5]}, {"1": 2, "2": [0.1, 0.9]}],
                      [{"1": 3, "2": [0.2, 0.8]}, {"1": 4, "2": [0.3, 0.7]}]]}"#,
        )
        .unwrap();
        let e = neg.read_checked(&q).unwrap_err();
        assert_eq!(
            (e.kind, e.variable.as_deref()),
            (ErrorKind::Data, Some("q:1"))
        );
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
