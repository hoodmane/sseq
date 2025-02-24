use fp::prime::ValidPrime;
use fp::vector::{Slice, SliceMut};

#[cfg(doc)]
use fp::vector::FpVector;

/// A graded algebra over $\mathbb{F}_p$.
///
/// Each degree is finite dimensional, and equipped with a distinguished ordered basis. Basis
/// elements are referred to by their signed degree and unsigned index, while a general
/// element of a given degree is denoted by an [`FpVector`] given in terms of that degree's
/// basis.
///
/// These algebras are frequently infinite-dimensional, so we must construct the representation
/// lazily. The function [`Algebra::compute_basis()`] will request that book-keeping information
/// be updated to perform computations up to the given degree; users must make sure to call
/// this function before performing other operations at that degree.
///
/// Algebras may have a distinguished set of generators; see [`GeneratedAlgebra`].
pub trait Algebra: std::fmt::Display + Send + Sync {
    /// A magic constant used to identify the algebra in save files. When working with the
    /// Milnor algebra, it is easy to forget to specify the algebra and load Milnor save files
    /// with the Adem basis. If we somehow manage to resume computation, this can have
    /// disasterous consequences. So we store the magic in the save files.
    ///
    /// This defaults to 0 for other kinds of algebra that don't care about this problem.
    fn magic(&self) -> u32 {
        0
    }

    /// Returns the prime the algebra is over.
    fn prime(&self) -> ValidPrime;

    /// Computes basis elements up to and including `degree`.
    ///
    /// This function must be called by users before other functions that will involve operations
    /// at `degree`, so it should be used to update internal data structure in perparation
    /// for such operations.
    ///
    /// This function must be idempotent and cheap to call again with the
    /// same argument.
    fn compute_basis(&self, degree: i32);

    /// Returns the dimension of the algebra in degree `degree`.
    fn dimension(&self, degree: i32) -> usize;

    /// Computes the product `r * s` of two basis elements, and adds the
    /// result to `result`.
    ///
    /// `result` is not required to be aligned.
    fn multiply_basis_elements(
        &self,
        result: SliceMut,
        coeff: u32,
        r_degree: i32,
        r_idx: usize,
        s_degree: i32,
        s_idx: usize,
    );

    /// Computes the product `r * s` of a basis element `r` and a general element `s`, and adds the
    /// result to `result`.
    ///
    /// Neither `result` nor `s` must be aligned.
    fn multiply_basis_element_by_element(
        &self,
        mut result: SliceMut,
        coeff: u32,
        r_degree: i32,
        r_idx: usize,
        s_degree: i32,
        s: Slice,
    ) {
        let p = self.prime();
        for (i, v) in s.iter_nonzero() {
            self.multiply_basis_elements(
                result.copy(),
                (coeff * v) % *p,
                r_degree,
                r_idx,
                s_degree,
                i,
            );
        }
    }

    /// Computes the product `r * s` of a general element `r` and a basis element `s`, and adds the
    /// result to `result`.
    ///
    /// Neither `result` nor `r` must be aligned.
    fn multiply_element_by_basis_element(
        &self,
        mut result: SliceMut,
        coeff: u32,
        r_degree: i32,
        r: Slice,
        s_degree: i32,
        s_idx: usize,
    ) {
        let p = self.prime();
        for (i, v) in r.iter_nonzero() {
            self.multiply_basis_elements(
                result.copy(),
                (coeff * v) % *p,
                r_degree,
                i,
                s_degree,
                s_idx,
            );
        }
    }

    /// Computes the product `r * s` of two general elements, and adds the
    /// result to `result`.
    ///
    /// Neither `result`, `s`, nor `r` must be aligned.
    fn multiply_element_by_element(
        &self,
        mut result: SliceMut,
        coeff: u32,
        r_degree: i32,
        r: Slice,
        s_degree: i32,
        s: Slice,
    ) {
        let p = self.prime();
        for (i, v) in s.iter_nonzero() {
            self.multiply_element_by_basis_element(
                result.copy(),
                (coeff * v) % *p,
                r_degree,
                r,
                s_degree,
                i,
            );
        }
    }

    /// Returns a list of filtration-one elements in $Ext(k, k)$.
    ///
    /// These are the same as indecomposable elements of the algebra.
    ///
    /// This function returns a default list of such elements in the format
    /// `(name, degree, index)` for which we want to compute products with in
    /// the resolutions.
    fn default_filtration_one_products(&self) -> Vec<(String, i32, usize)> {
        Vec::new()
    }

    /// Converts a basis element into a string for display.
    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String;

    /// Converts a general element into a string for display.
    fn element_to_string(&self, degree: i32, element: Slice) -> String {
        let mut result = String::new();
        let mut zero = true;
        for (idx, value) in element.iter_nonzero() {
            zero = false;
            if value != 1 {
                result.push_str(&format!("{} * ", value));
            }
            let b = self.basis_element_to_string(degree, idx);
            result.push_str(&format!("{} + ", b));
        }
        if zero {
            result.push('0');
        } else {
            // Remove trailing " + "
            result.pop();
            result.pop();
            result.pop();
        }
        result
    }
}

#[cfg(feature = "json")]
pub trait JsonAlgebra: Algebra {
    /// A name for the algebra to use in serialization operations.
    fn prefix(&self) -> &str;

    /// Parses a basis element from JSON.
    ///
    /// The representation is defined by the algebra itself, and is used by by
    /// module specifications to specify basis elements.
    fn json_to_basis(&self, json: &serde_json::Value) -> anyhow::Result<(i32, usize)>;

    /// Converts a basis element into JSON.
    fn json_from_basis(&self, degree: i32, idx: usize) -> serde_json::Value;
}

/// An [`Algebra`] equipped with a distinguished presentation.
///
/// These data can be used to specify finite modules as the actions of the distinguished generators.
pub trait GeneratedAlgebra: Algebra {
    /// Return generators in `degree`.
    ///
    /// Generators are specified as basis element indices in that degree. The order of the
    /// list is not important.
    ///
    /// This method need not be fast, because they will only be performed when constructing the module,
    /// and will often only involve low dimensional elements.
    fn generators(&self, degree: i32) -> Vec<usize>;

    /// Returns the name of a generator.
    ///
    /// Note: `idx` is the index within `degree`'s basis, *not* the list returned by
    /// [`GeneratedAlgebra::generators()`].
    ///
    /// By default, this function will forward to [`Algebra::basis_element_to_string()`], but
    /// may be overridden if more concise names are available.
    ///
    /// This function MUST be inverse to [`GeneratedAlgebra::string_to_generator()`].
    fn generator_to_string(&self, degree: i32, idx: usize) -> String {
        self.basis_element_to_string(degree, idx)
    }

    /// Parse `input` into a generator.
    ///
    /// This function is a `nom` combinator.
    ///
    /// This function MUST be inverse to `string_to_generator` (and not `basis_element_to_string`).
    fn string_to_generator<'a>(&self, input: &'a str) -> nom::IResult<&'a str, (i32, usize)>;

    /// Decomposes an element into generators.
    ///
    /// Given a basis element $A$, this function returns a list of triples
    /// $(c_i, A_i, B_i)$, where $A_i$ and $B_i$ are basis elements of strictly
    /// smaller degree than $A$, such that
    ///
    /// $$ A = \sum_i c_i A_i B_i.$$
    ///
    /// Combined with actions for generators, this allows us to recursively compute the action
    /// of an element on a module.
    ///
    /// This method need not be fast, because they will only be performed when constructing the module,
    /// and will often only involve low dimensional elements.
    fn decompose_basis_element(
        &self,
        degree: i32,
        idx: usize,
    ) -> Vec<(u32, (i32, usize), (i32, usize))>;

    /// Returns relations that the algebra wants checked to ensure the consistency of module.
    ///
    /// Relations are encoded as general multi-degree elements which are killed in the quotient:
    /// $$ \sum_i c_i \alpha_i \beta_i = 0. $$
    /// where $c_i$ are coefficients and $\alpha_i$ and $\beta_i$ are basis elements of
    /// arbitrary degree.
    fn generating_relations(&self, degree: i32) -> Vec<Vec<(u32, (i32, usize), (i32, usize))>>;
}

#[macro_export]
macro_rules! dispatch_algebra {
    ($struct:ty, $dispatch_macro: ident) => {
        impl Algebra for $struct {
            $dispatch_macro! {
                fn magic(&self) -> u32;
                fn prime(&self) -> ValidPrime;
                fn compute_basis(&self, degree: i32);
                fn dimension(&self, degree: i32) -> usize;
                fn multiply_basis_elements(
                    &self,
                    result: SliceMut,
                    coeff: u32,
                    r_degree: i32,
                    r_idx: usize,
                    s_degree: i32,
                    s_idx: usize,
                );

                fn multiply_basis_element_by_element(
                    &self,
                    result: SliceMut,
                    coeff: u32,
                    r_degree: i32,
                    r_idx: usize,
                    s_degree: i32,
                    s: Slice,
                );

                fn multiply_element_by_basis_element(
                    &self,
                    result: SliceMut,
                    coeff: u32,
                    r_degree: i32,
                    r: Slice,
                    s_degree: i32,
                    s_idx: usize,
                );

                fn multiply_element_by_element(
                    &self,
                    result: SliceMut,
                    coeff: u32,
                    r_degree: i32,
                    r: Slice,
                    s_degree: i32,
                    s: Slice,
                );

                fn default_filtration_one_products(&self) -> Vec<(String, i32, usize)>;

                fn basis_element_to_string(&self, degree: i32, idx: usize) -> String;

                fn element_to_string(&self, degree: i32, element: Slice) -> String;
            }
        }
    };
}
