/*
* Project: Recognizing Handwritten Digits
* Authors: Awildo G., Sosina A., Siqi F., Sam. A.
* Date: June 7, 2020
* Description: The construction of the handwriting-recognizing neural network as
*              described in Michael Nielsen's textbook on Introduction to 
*              Neural Networks and Deep Learning
* CITES: http://neuralnetworksanddeeplearning.com/index.html
*        https://docs.rs/rand/0.7.3/rand/seq/trait.SliceRandom.html#tymethod.shuffle
*
* To do: ✓✓✓✓✓✓✓✓✓✓✓ c✓✓ instead of c++
*      o Network struct
*      ✓ Constructor
*      ✓ feedforward
*      o update_mini_batch
*      o SGD
*      o backprop
*      ✓ sigmoid
*      ✓ sigmoid_prime
*      ✓ evaluate
*      ✓ cost_derivative
*/

mod rustify;
#[macro_use(c)]
extern crate cute;

struct Network {
    sizes: Vec<usize>,
    num_layers: usize,
    biases: Vec< rustify::Array2<f64> >,
    weights: Vec< rustify::Array2<f64> >,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        Self {
            sizes: sizes.clone(),
            num_layers: sizes.len(),
            biases: c![rustify::randn((*y, 1)), for y in sizes[1..].iter()],
            weights: c![rustify::randn((*x.1, *x.0)),
                        for x in sizes[..(sizes.len() - 1)]
                                       .iter().zip(sizes[1..].iter())],
        }
    }

    // pub fn print(&self) {
    //     println!("w: {:?}", self.weights);
    // }

    pub fn feedforward(&self, a: &mut rustify::Array2<f64>) -> rustify::Array2<f64> {
        let mut result: rustify::Array2<f64> = a.clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            result = sigmoid(& (w.dot(&result) + b) );
        } 
        result
    }

    // pub fn SGD(&mut self) {

    // }

    // pub fn update_mini_batch(&mut self) {

    // }

    pub fn backprop(& self, x: & rustify::Array2<f64>, y: usize) -> (Vec< rustify::Array2<f64> >, Vec< rustify::Array2<f64> >){
        let mut nabla_b = c![rustify::Array2::<f64>::zeros((b.shape()[0], b.shape()[1])), for b in self.biases.iter()];
        let mut nabla_w = c![rustify::Array2::<f64>::zeros((w.shape()[0], w.shape()[1])), for w in self.weights.iter()];
        
        let mut activation: rustify::Array2<f64> = x.clone();
        let mut activations = vec![activation.clone()];
        let mut zs: Vec< rustify::Array2<f64> > = Vec::new();

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            activation = sigmoid(&z);
            activations.push(activation.clone());
        }
        let delta = self.cost_derivative(&activations[activations.len() - 1], y) * 
                        sigmoid_prime(&zs[zs.len() - 1]);

        let len_b = nabla_b.len();
        let len_w = nabla_w.len(); 
        nabla_b[len_b - 1] = delta.clone();
        nabla_w[len_w - 1] = delta.dot(&activations[activations.len() - 2].t());

        // Converting negative indexing is ugly
        for layer in 2..self.num_layers {
            let len_z = zs.len();
            let len_activations = activations.len();
            let z = &zs[((len_z as isize) - (layer as isize)) as usize];
            let sp = sigmoid_prime(z);
            let idx_plus = ((len_w as isize) + (-1 * (layer as isize) + 1)) as usize; // -l + 1
            let idx_minus = ((len_activations as isize) + (-1 * (layer as isize) - 1)) as usize; // -l -1 
            let delta = self.weights[idx_plus].t().dot(&delta) * sp;
            nabla_b[((len_b as isize) - (layer as isize)) as usize] = delta.clone();
            nabla_w[((len_w as isize) - (layer as isize)) as usize] = delta.dot(&activations[idx_minus].t());
        }
        (nabla_b, nabla_w)
    }

    pub fn evaluate(&self, test_data: Vec<&mut (rustify::Array2<f64>, usize)>) -> usize {
        let test_results = c![(rustify::argmax(&self.feedforward(&mut x.0)), x.1), 
                                for x in test_data];
        test_results.iter().map(|w| (w.0 == w.1) as usize).sum()
    }

    pub fn cost_derivative(&self, output_activations: & rustify::Array2<f64>, y: usize) -> rustify::Array2<f64> {
        output_activations - (y as f64)
    }
}

/// Private auxillary functions

fn sigmoid(z: & rustify::Array2<f64>) -> rustify::Array2<f64> {
    1.0/(1.0 + rustify::exp(&z, -1.0))
}

// The first derivative of the sigmoid function 
fn sigmoid_prime(z: & rustify::Array2<f64>) -> rustify::Array2<f64> {
    sigmoid(z) * (1.0 - sigmoid(z))
}

fn main() {
    // println!("Hello, World!");
    // let n = Network::new(vec![2,3,4,5]);
    // n.print();

    let mut a1 = rustify::Array2::<f64>::zeros((3,1));
    a1[[0,0]] = 2.;
    a1[[1,0]] = 1.;
    a1[[2,0]] = -1.;

    println!("{:?}", sigmoid(&a1));
    println!("{:?}", sigmoid_prime(&a1));

}