/*
* Project: Recognizing Handwritten Digits
* Authors: Awildo G., Sosina A., Siqi F., Sam. A.
* Date: June 6, 2020
* Description: Module of auxillary functions that translates some important
*              numpy (python) functions to be used in the construction
*              of the handwriting-recognizing neural network
* CITES: https://docs.rs/rand/0.6.5/rand/distributions/struct.Normal.html
*        https://docs.rs/ndarray/0.13.1/ndarray/type.ArrayD.html
*/

// To do:
//      ✓ dot (Done using ndarray::Array2)
//      ✓ Hadamard product (Done using * from ndarray)
//      ✓ argmax
//      ✓ zeros (Done using ndarray::Array2)
//      ✓ exp - vectorization
//      ✓ randn
//      o zip (array1, array2) -> { (array1, array2) }
//      o network struct


pub use ndarray::Array2;                    // Allows us to use 2D Matrices
use rand::prelude::*;                       // Used in randn
use rand_distr::StandardNormal;             // Defines the type of random nums


// Takes in a tuple of usize, which represent the dimensions of the
// resulting array.
// Returns values an 2D matrix of type Array2<f64> with the given shape
// and the values initialized to random numbers drawn from the sample of the
// standard normal distribution (mean 0 & standard deviation 1).
pub fn randn((x, y): (usize, usize)) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((x, y));
    for row in 0..x {
        for col in 0..y {
            let val: f64 = thread_rng().sample(StandardNormal);
            result[[row, col]] = val
        }
    }
    result
}//----------------------------------------------------------------------------


// Let v = < v_1, v_2, ..., v_n>.
// Then, exp(v) = < exp(v_1), exp(v_2), ..., exp(v_n) > 
pub fn exp(z: & Array2<f64>) -> Array2<f64> {
    // Euler's constant
    const E: f64 = 2.71828182845904523536028747135266250f64;

    let z_sh = z.shape();
    let mut result = Array2::<f64>::zeros((z_sh[0], z_sh[1]));
    for (idx, row) in z.iter().enumerate() {
        // only works for vectors
        result[[idx, 0]] = E.powf(*row);
    }
    result
}//----------------------------------------------------------------------------

// argmax is for an Array2
// take each row in the matrix, add it to a random vector, find the largest of
// that
// argmax takes a matrix and outputs the index/location of the maximum
// element in that matrix
// 0 0 0
// 0 1 0   -> 4
// 0 0 0
pub fn argmax(mat: &Array2<f64>) -> usize {
    // We will flatten the Array2
    // Go through each row, keep track of the maximum index 
    let mut flat_mat: Vec<f64> = vec![];
    for row in mat.iter() {
        // row is of type &T
        flat_mat.push(*row);
    }

    // At this point, flat_mat is a vector of the rows of mat1

    let mut argmax: usize = 0;
    let mut max: f64 = flat_mat[0];
    for (idx, val) in flat_mat.iter().enumerate(){
        if max < *val {
            max = *val;
            argmax = idx;
        }
    }
    argmax
}//----------------------------------------------------------------------------

// Used only for debugging auxillary functions
// fn main() {
    // Array2::<type>::zeros(shape);
    // let mut _a_2 = Array2::<f64>::zeros((2,3));
    // let mut _b_2 = Array2::<f64>::zeros((3,1));
    // let mut _c_2 = Array2::<f64>::zeros((3,1));

//-----------------------------------------------------------------------------
    /*    Indexing the 2D Matrix    */
    // _c_2[[row, col]] for accessing

    // _b_2[[0,0]] = 1.;
    // _b_2[[1,0]] = 2.;
    // _b_2[[2,0]] = 3.;
    //------------------
    // _c_2[[0,0]] = 1.;
    // _c_2[[1,0]] = 2.;
    // _c_2[[2,0]] = 3.;
//-----------------------------------------------------------------------------
    /* Testing the methods we wrote */  
    // println!("{}", randn((3,2)));
    // println!("{}, {}", argmax(&_b_2), _b_2);
    // println!("{:?}", exp(& _b_2));
    // let _c = _a_2.dot(&_b_2);
    // let _c = &_c_2 * &_b_2; // 3 x 1 zero matrix
    // println!("{:?}", _c);
// }
