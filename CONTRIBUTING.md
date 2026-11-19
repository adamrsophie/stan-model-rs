# Contributing

* Run `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test --workspace` before sending a change; CI runs the same on Linux, macOS, and Windows.
* Every numerical feature needs a test against an independent reference: a closed form, CmdStan,
  finite differences, or a round trip. A test that only calls a function is not enough.
* Mathematics belongs in stanmath-rs. The runtime composes transforms and densities; it does not
  implement them.
* No `unsafe` code (the workspace forbids it).
