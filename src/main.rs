///
/// Project: Recognizing Handwritten Digits
/// Authors: Awildo G., Sosina A., Siqi F., Sam. A.
/// Date: June 6, 2020
/// Description: Implementation of network.py file from Ian Goodfellow's
///              textbook on neural networks in Rust.
/// 
/// To do:  
///        ✓ dot (Done using ndarray::Array2)
///        ✓ Hadamard product (Done using * from ndarray)
///        ✓ argmax 
///        ✓ zeros (Done using ndarray::Array2)
///        ✓ exp - vectorization
///        ✓ randn
///        o zip (array1, array2) -> { (array1, array2) }
///        o network struct
/// 
/// 
/// We used:
/// https://docs.rs/rand/0.6.5/rand/distributions/struct.Normal.html
/// https://docs.rs/ndarray/0.13.1/ndarray/type.ArrayD.html
/// 
// / 

use ndarray::Array2;
use rand::prelude::*;
use rand_distr::StandardNormal as norm;

// mean 2, standard deviation 3
// let normal = Normal::new(2.0, 3.0);
// let v = normal.sample(&mut rand::thread_rng());
// println!("{} is from a N(2, 9) distribution", v)

fn randn((x, y): (usize, usize)) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((x, y));
    for row in 0..x {
        for col in 0..y {
            let val: f64 = thread_rng().sample(norm);
            result[[row, col]] = val
        }
    }
    result
}

pub const E: f64 = 2.71828182845904523536028747135266250f64; 
// Euler's constant 

// Let v = < v_1, v_2, ..., v_n>.
// Then, exp(v) = < exp(v_1), exp(v_2), ..., exp(v_n) > 
fn exp(z: & Array2<f64>) -> Array2<f64> {
    let z_sh = z.shape();
    let mut result = Array2::<f64>::zeros((z_sh[0], z_sh[1]));
    for (idx, row) in z.iter().enumerate() {
        // only works for vectors
        result[[idx, 0]] = E.powf(*row);
    }
    result
}

// we want to call as such: 
// test_results = [(np.argmax(self.feedforward(x)), y)
//                        for (x, y) in test_data]

// argmax is for an Array2
// take each row in the matrix, add it to a random vector, find the largest of that

// Auxillary function used in evaluate
fn argmax(mat: &Array2<f64>) -> usize {
    // 0 0 0
    // 0 1 0   -> 4
    // 0 0 0

    // argmax takes a matrix and outputs the index/location of the maximum
    // element in that matrix
 
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
}

fn main() {
    // let a = array![
    //             [1.,2.,3.],
    //             [4.,5.,6.],
    //         ];
    // assert_eq!(a.ndim(), 2);         // get the number of dimensions of array a
    // assert_eq!(a.len(), 6);          // get the number of elements in array a
    // assert_eq!(a.shape(), [2, 3]);   // get the shape of array a
    // assert_eq!(a.is_empty(), false); // check if the array has zero elements


    // Create a 2 × 3 array using the dynamic dimension type
    // let mut a = ArrayD::<f64>::zeros(IxDyn(&[2, 3]));
    // Create a 3 × 5 array using the dynamic dimension type
    // IxDyn takes size of array not index
    // let mut b = ArrayD::<f64>::zeros(IxDyn(&[3, 5]));

    // We can use broadcasting to add arrays of compatible shapes together:
    // a += &b;
    // println!("{:?}", a);

    // We can index into a, b using fixed size arrays:
    // a[[0, 0, 0]] = 2.;
    // a[[1, 1, 0]] = 3.;
    // b[[1, 2, 3]] = a[[1, 1, 0]];
    // Note: indexing will panic at runtime if the number of indices given does
    // not match the array.

    // let ziptest = a.iter().zip(b.iter());
    // // println!("{:?}", ziptest);
    // for (x, y) in ziptest {
    //     println!("{}, {}", x, y);
    // }
    // println!("{:?}", ziptest.next());
    
    //86-108
    
    // Array2::<type>::zeros(shape);
    // let mut _a_2 = Array2::<f64>::zeros((2,3));
    let mut _b_2 = Array2::<f64>::zeros((3,1));
    // // _b_2 = [[0],
    // //         [0],
    // //         [0]]
    // let mut _c_2 = Array2::<f64>::zeros((3,1));

    // // _c_2[[row, col]] for accessing
    _b_2[[0,0]] = 1.;
    _b_2[[1,0]] = 2.;
    _b_2[[2,0]] = 3.;

    println!("{}", randn((3,2)));

    // println!("{}, {}", argmax(&_b_2), _b_2);
    // println!("{:?}", exp(& _b_2));
    
    // _c_2[[0,0]] = 1.;
    // _c_2[[1,0]] = 2.;
    // _c_2[[2,0]] = 3.;

    // // let _c = _a_2.dot(&_b_2);
    // let _c = &_c_2 * &_b_2; // 3 x 1 zero matrix
    
    // // We can keep them in the same vector because both the arrays have
    // // the same type `Array<f64, IxDyn>` a.k.a `ArrayD<f64>`:
    // let _arrays = vec![_a_2, _b_2];
    // println!("{:?}", _c);

    // DOT PRODUCT TESTING 
    // a.dot(&b);
}
