/*
* Project: Recognizing Handwritten Digits
* Authors: Awildo G., Sosina A., Siqi F., Sam. A.
* Date: June 7, 2020
* Description: The construction of the handwriting-recognizing neural network as
*              described in Michael Nielsen's textbook on Introduction to 
*              Neural Networks and Deep Learning
* CITES: http://neuralnetworksanddeeplearning.com/index.html
*/

mod rustify;

fn main() {
    println!("Hello, World!");

    // accessing use ndarray::Array2;
    let mut a = rustify::Array2::<f64>::zeros((3,1));
    a[[0,0]] = 2.;
    a[[1,0]] = 4.;
    a[[2,0]] = 5.;
    // accessing methods from rustify
    println!("{:?}\n\n", rustify::exp(&a));

    let b = rustify::randn((5,3));
    println!("The matrix: {:?} has size: {:?}", b, b.shape());
}