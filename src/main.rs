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
* To do: 
*      ✓ Network struct
*      ✓ Constructor
*      ✓ feedforward
*      ✓ update_mini_batch
*      ✓ SGD
*      ✓ backprop
*      ✓ sigmoid
*      ✓ sigmoid_prime
*      ✓ evaluate
*      ✓ cost_derivative
*/

mod rustify;
#[macro_use(c)]
extern crate cute;
use rand::thread_rng;
use rand::seq::SliceRandom;

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

    pub fn feedforward(&self, a: rustify::Array2<f64>) -> rustify::Array2<f64> {
        let mut result: rustify::Array2<f64> = a.clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            result = sigmoid(& (w.dot(&result) + b) );
        }
        result
    }

    pub fn SGD(&mut self, training_data: &mut Vec<(rustify::Array2<f64>, usize)>, epochs: usize, mini_batch_size: usize, eta: f64, test_data: Option<Vec<&mut (rustify::Array2<f64>, usize)>>) {
        let mut n_test: usize = 0;
        let mut test_data2: Vec<&mut (rustify::Array2<f64>, usize)> = vec![];
        if let Some(i) = test_data { // SOME (test_data) alternatively
            n_test = i.len();
            test_data2 = i;
        }

        let n = training_data.len();
        for j in 0..epochs {
            // shuffle the training data
            training_data.shuffle(&mut thread_rng());
            let mut mini_batches: Vec< Vec< (rustify::Array2<f64>, usize) > > = vec![];
            for k in (0..n).step_by(mini_batch_size) {
                if k + mini_batch_size < n {
                    let mut batch: Vec<(rustify::Array2<f64>, usize)> = vec![];
                    for l in k..(k + mini_batch_size) {
                        batch.push(training_data[l].clone());
                    }
                    mini_batches.push(batch);
                }
                else {
                    let mut batch: Vec<(rustify::Array2<f64>, usize)> = vec![];
                    let mut i = k;
                    while i < n {
                        batch.push(training_data[i].clone());
                        i += 1;
                    }
                    mini_batches.push(batch);
                }
            }

            for mini_batch in mini_batches {
                self.update_mini_batch(& mini_batch, eta);
            }
            if n_test != 0 {
                println!("Epoch {}: {} / {}", j, self.evaluate(& test_data2), n_test);
            }
            else {
                println!("Epoch {} complete", j);
            }
        }  
    }

    pub fn update_mini_batch(&mut self, mini_batch: & Vec<(rustify::Array2<f64>, usize)>, eta: f64) {
        let mut nabla_b = c![rustify::Array2::<f64>::zeros((b.shape()[0], b.shape()[1])), for b in self.biases.iter()];
        let mut nabla_w = c![rustify::Array2::<f64>::zeros((w.shape()[0], w.shape()[1])), for w in self.weights.iter()];
    
        for (x, y) in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(&x, *y);
            nabla_b = c![nabla.0 + nabla.1, for nabla in nabla_b.iter().zip(delta_nabla_b.iter())];
            nabla_w = c![nabla.0 + nabla.1, for nabla in nabla_w.iter().zip(delta_nabla_w.iter())];
        }

        let mut weights2: Vec< rustify::Array2<f64> > = vec![];
        for (w, nw) in self.weights.iter().zip(nabla_w.iter()) {
            let w2: rustify::Array2<f64> = w.clone();
            let nw2: rustify::Array2<f64> = nw.clone();
            let math_part = w2 - ((eta/(mini_batch.len() as f64)) * nw2);
            weights2.push(math_part);
        }
        self.weights = weights2;
    
        let mut biases2: Vec< rustify::Array2<f64> > = vec![];
        for (b, nb) in self.biases.iter().zip(nabla_b.iter()) {
            let b2: rustify::Array2<f64> = b.clone();
            let nw2: rustify::Array2<f64> = nb.clone();
            let math_part = b2 - ((eta/(mini_batch.len() as f64)) * nw2);
            biases2.push(math_part);
        }
        self.biases = biases2;


        // self.weights = c![wt.0 - (wt.1 * (eta/(mini_batch.len() as f64))), for wt in self.weights.iter().zip(nabla_w.iter())];
        // self.biases = c![bs.0  - ((eta/ (mini_batch.len() as f64))*bs.1), for bs in self.biases.iter().zip(nabla_b.iter())];
    }

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

    pub fn evaluate(&self, test_data: & Vec<&mut (rustify::Array2<f64>, usize)>) -> usize {
        let test_results = c![(rustify::argmax(&self.feedforward(x.0.clone())), x.1), 
                                for x in test_data];
        test_results.iter().map(|w| (w.0 == w.1) as usize).sum()
    }

    pub fn cost_derivative(&self, output_activations: & rustify::Array2<f64>, y: usize) -> rustify::Array2<f64> {
        output_activations - (y as f64)
    }
}

/// Private auxillary functions

pub fn sigmoid(z: & rustify::Array2<f64>) -> rustify::Array2<f64> {
    1.0/(1.0 + rustify::exp(&z, -1.0))
}

// The first derivative of the sigmoid function 
pub fn sigmoid_prime(z: & rustify::Array2<f64>) -> rustify::Array2<f64> {
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