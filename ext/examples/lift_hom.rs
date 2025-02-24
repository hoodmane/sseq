//! Given an element in $\Ext(M, N)$, this computes the induced map $\Ext(N, k) \to \Ext(M, k)$
//! given by composition.
//!
//! It begins by asking for the two modules $M$, $N$ and the $\Ext$ class. Afterwards, you may
//! either supply save files for the two modules, or a range to compute the map for.
//!
//! Afterwards, the user is prompted for the Ext class. If $R_s$ is the $s$th term of the minimal
//! resolution of $M$, the Ext class is given as an element of $\Hom_A(R_s, \Sigma^t N) =
//! \Hom(\Ext^{s, *}(M, k)^\vee, \Sigma^t N)$.
//!
//! In other words, for every basis element in $\Ext^{s, *}(M, k)$, one has to specify its image in
//! $\Sigma^t N$. In the special case where $s = 0$, this is specifying the map between the
//! underlying modules on module generators under the Steenrod action.
//!
//! Our notation is as follows:
//!
//!  - `f` is the map in $\Hom_A(R_s, \Sigma^t N)$.
//!  - `F` is the induced map on Ext.
//!
//! Each prompt will be of the form `f(x_(s, n, i)) = ` and the user has to input the value of the
//! homomorphism on this basis element. For example, the following session computes the map induced
//! by the projection of spectra $C2 \to S^1$
//!
//! ```text
//!  $ cargo run --example lift_hom
//! Target module (default: S_2): C2
//! Source module (default: Cnu): S_2
//! s of Ext class (default: 0): 0
//! n of Ext class (default: 0): -1
//! Target save file (optional):
//! Max target s (default: 10): 10
//! Max target n (default: 10): 20
//!
//! Input module homomorphism to lift:
//! f(x_(0, 0, 0)): [1]
//! ```
//!
//! It is important to keep track of varaince when using this example; Both $\Ext(-, k)$ and
//! $H^*(-)$ are contravariant functors. The words "source" and "target" refer to the map between
//! Steenrod modules.

use algebra::module::{BoundedModule, Module};
use ext::chain_complex::{AugmentedChainComplex, ChainComplex, FreeChainComplex};
use ext::resolution_homomorphism::ResolutionHomomorphism;
use ext::utils;
use fp::matrix::Matrix;

use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let target = {
        let mut target = utils::query_module_only("Target module", None)?;
        target.load_quasi_inverse = target.save_dir().is_none();
        Arc::new(target)
    };

    let source_equal_target = query::yes_no("Source equal to target?");
    let source = if source_equal_target {
        Arc::clone(&target)
    } else {
        let mut s = utils::query_module_only("Source module", None)?;
        s.load_quasi_inverse = false;
        Arc::new(s)
    };

    assert_eq!(source.prime(), target.prime());
    let p = source.prime();

    let name: String = query::raw("Name of product", str::parse);

    let shift_n: i32 = query::with_default("n of Ext class", "0", str::parse);
    let shift_s: u32 = query::with_default("s of Ext class", "0", str::parse);
    let shift_t = shift_n + shift_s as i32;

    let n: i32 = query::with_default("Max target n", "10", str::parse);
    let s: u32 = query::with_default("Max target s", "10", str::parse);

    if source_equal_target {
        target.compute_through_stem(s + shift_s, n + std::cmp::max(0, shift_n));
    } else {
        source.compute_through_stem(s + shift_s, n + shift_n);
        target.compute_through_stem(s, n);
    }

    let target_module = target.target().module(0);
    let hom = ResolutionHomomorphism::new(name.clone(), source, target, shift_s, shift_t);

    eprintln!("\nInput Ext class to lift:");
    for output_t in 0..=target_module.max_degree() {
        let input_t = output_t + shift_t;
        let mut matrix = Matrix::new(
            p,
            hom.source.number_of_gens_in_bidegree(shift_s, input_t),
            target_module.dimension(output_t),
        );

        if matrix.rows() == 0 || matrix.columns() == 0 {
            hom.extend_step(shift_s, input_t, None);
        } else {
            for (idx, row) in matrix.iter_mut().enumerate() {
                let v: Vec<u32> =
                    query::vector(&format!("f(x_({shift_s}, {input_t}, {idx}))"), row.len());
                for (i, &x) in v.iter().enumerate() {
                    row.set_entry(i, x);
                }
            }
            hom.extend_step(shift_s, input_t, Some(&matrix));
        }
    }

    hom.extend_all();

    for (s, n, t) in hom.target.iter_stem() {
        if s + shift_s >= hom.source.next_homological_degree()
            || t + shift_t > hom.source.module(s + shift_s).max_computed_degree()
        {
            continue;
        }
        let matrix = hom.get_map(s + shift_s).hom_k(t);
        for (i, r) in matrix.iter().enumerate() {
            println!("{name} x_({n}, {s}, {i}) = {r:?}");
        }
    }
    Ok(())
}
