//! Random number generation.
//!
//! The runtime never owns a global generator. Model code receives an explicit `&mut dyn Rng`:
//! transformed data draws from a generator the model constructor seeds, and generated
//! quantities draw from the generator the caller passes to [`crate::Model::write_array`]. The
//! services layer owns the streams and decides when generated quantities run.
//!
//! The generator is [`StdRng`] from the `rand` crate (currently ChaCha12). Draws are reproducible
//! for a fixed seed and `rand` version, but they are **not** bit-for-bit identical to CmdStan's,
//! which uses boost's `ecuyer1988` generator and its own sampling algorithms.

use rand::SeedableRng;
pub use rand::rngs::StdRng;

/// A generator for stream `stream` of seed `seed`.
///
/// For one `seed`, distinct `stream` numbers give distinct 64-bit generator seeds (the stream is
/// mixed in with SplitMix64, a bijection), so chains, transformed data, and generated quantities
/// can use separate streams of one user seed without sharing state.
pub fn stream(seed: u64, stream: u64) -> StdRng {
    let mixed = splitmix64(seed ^ splitmix64(stream.wrapping_add(0x9E37_79B9_7F4A_7C15)));
    StdRng::seed_from_u64(mixed)
}

/// Stream number used for transformed data, by convention.
pub const TRANSFORMED_DATA_STREAM: u64 = u64::MAX;

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn reproducible_and_independent() {
        let a: Vec<u64> = (0..4)
            .map(|_| 0)
            .scan(stream(42, 1), |r, _| Some(r.next_u64()))
            .collect();
        let b: Vec<u64> = (0..4)
            .map(|_| 0)
            .scan(stream(42, 1), |r, _| Some(r.next_u64()))
            .collect();
        let c: Vec<u64> = (0..4)
            .map(|_| 0)
            .scan(stream(42, 2), |r, _| Some(r.next_u64()))
            .collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
