#![allow(dead_code)]
use libc::{c_char, c_double, c_float, c_int};

mod ffi {
    use super::*;
    extern "C" {
        pub fn vsTanh(n: c_int, a: *const c_float, y: *mut c_float);
        pub fn vdTanh(n: c_int, a: *const c_double, y: *mut c_double);

        pub fn vs_add(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
        pub fn vd_add(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);
        pub fn vs_sub(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
        pub fn vd_sub(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);
        pub fn vs_mul(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
        pub fn vd_mul(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);
        pub fn vs_div(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
        pub fn vd_div(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);

        pub fn sgemm_(
            transa: *const c_char,
            transb: *const c_char,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            alpha: *const c_float,
            a: *const c_float,
            lda: *const c_int,
            b: *const c_float,
            ldb: *const c_int,
            beta: *const c_float,
            c: *mut c_float,
            ldc: *const c_int,
        );
        pub fn dgemm_(
            transa: *const c_char,
            transb: *const c_char,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            alpha: *const c_double,
            a: *const c_double,
            lda: *const c_int,
            b: *const c_double,
            ldb: *const c_int,
            beta: *const c_double,
            c: *mut c_double,
            ldc: *const c_int,
        );
        pub fn hgemm_(
            transa: *const c_char,
            transb: *const c_char,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            alpha: *const half::f16,
            a: *const half::f16,
            lda: *const c_int,
            b: *const half::f16,
            ldb: *const c_int,
            beta: *const half::f16,
            c: *mut half::f16,
            ldc: *const c_int,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn sgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    ffi::sgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn dgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    ffi::dgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn hgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: half::f16,
    a: &[half::f16],
    lda: i32,
    b: &[half::f16],
    ldb: i32,
    beta: half::f16,
    c: &mut [half::f16],
    ldc: i32,
) {
    ffi::hgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[inline]
fn vs_tanh(a: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let y_len = y.len();
    if a_len != y_len {
        panic!("a and y have different lengths {a_len} <> {y_len}")
    }
    unsafe { ffi::vsTanh(a_len as i32, a.as_ptr(), y.as_mut_ptr()) }
}

#[inline]
fn vd_tanh(a: &[f64], y: &mut [f64]) {
    let a_len = a.len();
    let y_len = y.len();
    if a_len != y_len {
        panic!("a and y have different lengths {a_len} <> {y_len}")
    }
    unsafe { ffi::vdTanh(a_len as i32, a.as_ptr(), y.as_mut_ptr()) }
}

// The vector functions from mkl can be performed in place by using the same array for input and
// output.
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/vector-mathematical-functions.html
#[inline]
pub fn vs_tanh_inplace(y: &mut [f32]) {
    unsafe { ffi::vsTanh(y.len() as i32, y.as_ptr(), y.as_mut_ptr()) }
}

#[inline]
pub fn vd_tanh_inplace(y: &mut [f64]) {
    unsafe { ffi::vdTanh(y.len() as i32, y.as_ptr(), y.as_mut_ptr()) }
}

#[inline]
pub fn vs_gelu(vs: &[f32], ys: &mut [f32]) {
    for (&v, y) in vs.iter().zip(ys.iter_mut()) {
        *y = (2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)
    }
    vs_tanh_inplace(ys);
    for (&v, y) in vs.iter().zip(ys.iter_mut()) {
        *y = 0.5 * v * (1.0 + *y)
    }
}

#[inline]
pub fn vd_gelu(vs: &[f64], ys: &mut [f64]) {
    for (&v, y) in vs.iter().zip(ys.iter_mut()) {
        *y = (2.0f64 / std::f64::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)
    }
    vd_tanh_inplace(ys);
    for (&v, y) in vs.iter().zip(ys.iter_mut()) {
        *y = 0.5 * v * (1.0 + *y)
    }
}

macro_rules! binary_op {
    ($fn_name:ident, $ty:ty) => {
        #[inline]
        pub fn $fn_name(a: &[$ty], b: &[$ty], y: &mut [$ty]) {
            let a_len = a.len();
            let b_len = b.len();
            let y_len = y.len();
            if a_len != y_len || b_len != y_len {
                panic!(
                    "{} a,b,y len mismatch {a_len} {b_len} {y_len}",
                    stringify!($fn_name)
                );
            }
            unsafe { ffi::$fn_name(a_len as i32, a.as_ptr(), b.as_ptr(), y.as_mut_ptr()) }
        }
    };
}
binary_op!(vs_add, f32);
binary_op!(vd_add, f64);
binary_op!(vs_sub, f32);
binary_op!(vd_sub, f64);
binary_op!(vs_mul, f32);
binary_op!(vd_mul, f64);
binary_op!(vs_div, f32);
binary_op!(vd_div, f64);
