//! Computes products in $\Mod_{C\tau^2}$.
//!
//! # Usage
//! The program asks for a module $M$ and an element $x \in \Ext^{\*, \*}(M, k)$. It then
//! computes the secondary product of the standard lift of $x$ with all (standard lifts of)
//! elements in $\Ext^{\*, \*}(M, k)$ that survive $d_2$.
//!
//! These products are computed for all elements whose product with $x$ lies in the specified
//! bidegree of $M$, and $k$ is resolved as far as necessary to support this computation.
//!
//! Running this program requires computing the secondary resolution of both $M$ and $k$, i.e. the
//! calculations performed by [`secondary`](../secondary/index.html). The user is encouraged to
//! make use of a save file to reuse these calculations for different products. (When $M$ is not
//! equal to $k$, the user will be prompted for the save directory of $k$)
//!
//! # Notes
//! The program verifies that $x$ is indeed permanent.

use std::path::PathBuf;
use std::sync::Arc;

use algebra::module::Module;
use fp::matrix::Matrix;
use fp::vector::FpVector;

use ext::chain_complex::{AugmentedChainComplex, ChainComplex, FreeChainComplex};
use ext::resolution_homomorphism::ResolutionHomomorphism;
use ext::secondary::*;
use ext::utils::query_module;

use itertools::Itertools;

fn main() -> anyhow::Result<()> {
    let resolution = Arc::new(query_module(
        Some(algebra::AlgebraType::Milnor),
        ext::utils::LoadQuasiInverseOption::IfNoSave,
    )?);

    let is_unit = resolution.target().modules.len() == 1 && resolution.target().module(0).is_unit();

    let unit = if is_unit {
        Arc::clone(&resolution)
    } else {
        let save_dir = query::optional("Unit save directory", |x| {
            core::result::Result::<PathBuf, std::convert::Infallible>::Ok(PathBuf::from(x))
        });
        Arc::new(ext::utils::construct("S_2@milnor", save_dir)?)
    };

    if !can_compute(&resolution) {
        eprintln!(
            "Cannot compute d2 for the module {}",
            resolution.target().module(0)
        );
        return Ok(());
    }

    let p = resolution.prime();

    let name: String = query::raw("Name of product", str::parse);

    let shift_n: i32 = query::raw(&format!("n of Ext class {name}"), str::parse);
    let shift_s: u32 = query::raw(&format!("s of Ext class {name}"), str::parse);
    let shift_t = shift_n + shift_s as i32;

    let hom = ResolutionHomomorphism::new(
        name,
        Arc::clone(&resolution),
        Arc::clone(&unit),
        shift_s,
        shift_t,
    );

    let mut matrix = Matrix::new(
        p,
        hom.source.number_of_gens_in_bidegree(shift_s, shift_t),
        1,
    );

    if matrix.rows() == 0 || matrix.columns() == 0 {
        panic!("No classes in this bidegree");
    }
    let v: Vec<u32> = query::vector("Input ext class", matrix.rows());
    for (i, &x) in v.iter().enumerate() {
        matrix[i].set_entry(0, x);
    }

    if !is_unit {
        unit.compute_through_stem(
            resolution.next_homological_degree() - 1 - shift_s,
            resolution.module(0).max_computed_degree() - shift_n,
        );
    }

    hom.extend_step(shift_s, shift_t, Some(&matrix));
    hom.extend_all();

    let res_lift = SecondaryResolution::new(Arc::clone(&resolution));
    res_lift.extend_all();

    // Check that class survives to E3.
    {
        let m = res_lift.homotopy(shift_s + 2).homotopies.hom_k(shift_t);
        assert_eq!(m.len(), v.len());
        let mut sum = vec![0; m[0].len()];
        for (x, d2) in v.iter().zip_eq(&m) {
            sum.iter_mut().zip_eq(d2).for_each(|(a, b)| *a += x * b);
        }
        assert!(
            sum.iter().all(|x| x % *p == 0),
            "Class supports a non-zero d2"
        );
    }
    let res_lift = Arc::new(res_lift);

    let unit_lift = if is_unit {
        Arc::clone(&res_lift)
    } else {
        let lift = SecondaryResolution::new(Arc::clone(&unit));
        lift.extend_all();
        Arc::new(lift)
    };

    let hom = Arc::new(hom);
    let hom_lift = SecondaryResolutionHomomorphism::new(
        Arc::clone(&res_lift),
        Arc::clone(&unit_lift),
        Arc::clone(&hom),
    );

    let start = std::time::Instant::now();

    hom_lift.extend_all();

    eprintln!("Time spent: {:?}", start.elapsed());

    // Compute E3 page
    let res_sseq = Arc::new(res_lift.e3_page());
    let unit_sseq = if is_unit {
        Arc::clone(&res_sseq)
    } else {
        Arc::new(unit_lift.e3_page())
    };

    fn get_page_data(sseq: &sseq::Sseq<sseq::Adams>, n: i32, s: u32) -> &fp::matrix::Subquotient {
        let d = sseq.page_data(n, s as i32);
        &d[std::cmp::min(3, d.len() - 1)]
    }

    let name = hom_lift.name();
    // Iterate through the multiplicand
    for (s, n, t) in unit.iter_stem() {
        // The potential target has to be hit, and we need to have computed (the data need for) the
        // d2 that hits the potential target.
        if !resolution.has_computed_bidegree(s + shift_s + 1, t + shift_t + 1) {
            continue;
        }
        if !resolution.has_computed_bidegree(s + shift_s - 1, t + shift_t) {
            continue;
        }

        let page_data = get_page_data(&*unit_sseq, n, s);

        if page_data.subspace_dimension() == 0 {
            continue;
        }

        let target_num_gens = resolution.number_of_gens_in_bidegree(s + shift_s, t + shift_t);
        let tau_num_gens = resolution.number_of_gens_in_bidegree(s + shift_s + 1, t + shift_t + 1);

        if target_num_gens == 0 && tau_num_gens == 0 {
            continue;
        }

        let mut outputs =
            vec![FpVector::new(p, target_num_gens + tau_num_gens); page_data.subspace_dimension()];

        hom_lift.hom_k(
            Some(&res_sseq),
            s,
            t,
            page_data.subspace_gens().map(|x| x.as_slice()),
            outputs.iter_mut().map(|x| x.as_slice_mut()),
        );
        for (gen, output) in page_data.subspace_gens().zip_eq(outputs) {
            print!("{name} [");
            ext::utils::print_element(gen.as_slice(), n, s);
            println!(
                "] = {} + τ {}",
                output.slice(0, target_num_gens),
                output.slice(target_num_gens, target_num_gens + tau_num_gens)
            );
        }
    }
    Ok(())
}
