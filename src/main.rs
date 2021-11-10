use ndarray::{prelude::*, Array};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::{
    convert::TryInto,
    f64::consts::E,
    fs::{self, File},
    io::{Read, Write},
    u32, usize,
};

struct TrainingData {
    input: Array1<f64>,
    expected: Array1<f64>,
}

struct FeedForward {
    layers: Vec<HiddenLayer>,
}

struct HiddenLayer {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

struct LayerResult {
    a: Array1<f64>,
    z: Array1<f64>,
}

struct LayerNabla {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl FeedForward {
    fn new(num_inputs: usize, num_nodes: usize, num_layers: usize, num_outputs: usize) -> Self {
        let mut instance = Self {
            layers: Vec::with_capacity(num_layers),
        };

        instance
            .layers
            .push(HiddenLayer::new(num_inputs, num_nodes));
        for _ in 1..num_layers {
            instance.layers.push(HiddenLayer::new(num_nodes, num_nodes));
        }
        instance
            .layers
            .push(HiddenLayer::new(num_nodes, num_outputs));

        instance
    }

    fn is_correct_result(&self, activation: &Array1<f64>, expecteds: &Array1<f64>) -> bool {
        let (mut highest_a, mut prediction) = (0.0_f64, 0);
        for (i, a) in activation.iter().enumerate() {
            if *a > highest_a {
                highest_a = *a;
                prediction = i;
            }
        }

        let mut expected = 0;
        for (i, e) in expecteds.iter().enumerate() {
            if *e == 1.0 {
                expected = i;
                break;
            }
        }
        prediction == expected
    }

    fn apply(&self, input: &Array1<f64>) -> Vec<LayerResult> {
        let mut layers_result = Vec::with_capacity(self.layers.len() + 1);
        layers_result.push(LayerResult {
            a: input.clone(),
            z: input.clone(),
        });

        for (i, layer) in self.layers.iter().enumerate() {
            let zs = layer.apply(&layers_result[i].a);
            layers_result.push(LayerResult {
                a: zs.mapv(|z| self.sigma(z)), // a⁽ᴸ⁾ = σ(z⁽ᴸ⁾)
                z: zs,                         // z⁽ᴸ⁾ = w⁽ᴸ⁾.a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾
            });
        }

        layers_result
    }

    fn train(&mut self, batch: &mut Vec<TrainingData>, eta: f64) {
        let mut total_nabla: Vec<_> = self
            .layers
            .iter()
            .map(|layer| LayerNabla {
                weights: Array2::zeros(layer.weights.raw_dim()),
                bias: Array1::zeros(layer.bias.raw_dim()),
            })
            .collect();

        batch.shuffle(&mut thread_rng());

        let mut total_cost = 0.;
        let mut score = 0;
        for (mbi, mini_batch) in batch.chunks(100).enumerate() {
            for training_data in mini_batch {
                let network_result = self.apply(&training_data.input);

                let training_cost =
                    self.cost(&network_result.last().unwrap().a, &training_data.expected);
                let training_prediction = self
                    .is_correct_result(&network_result.last().unwrap().a, &training_data.expected);

                let training_nabla = self.back_propagate(network_result, training_data);
                for (nabla, training_layer_nabla) in
                    total_nabla.iter_mut().rev().zip(training_nabla)
                {
                    nabla.weights += &training_layer_nabla.weights;
                    nabla.bias += &training_layer_nabla.bias;
                }

                total_cost += training_cost;
                score += if training_prediction { 1 } else { 0 };
            }
            print!("\rScore: {:6} / {:6}", score, mbi * mini_batch.len());

            for (layer, nabla) in self.layers.iter_mut().zip(&total_nabla) {
                layer.weights -= &(&nabla.weights * eta / mini_batch.len() as f64);
                layer.bias -= &(&nabla.bias * eta / mini_batch.len() as f64);
            }
        }
        println!();
        println!("Cost: {}", total_cost / batch.len() as f64);
    }

    fn back_propagate(
        &self,
        mut network_result: Vec<LayerResult>,
        training_data: &TrainingData,
    ) -> Vec<LayerNabla> {
        let last_result = network_result.pop().unwrap();

        // 𝛿⁽ᴸ⁾ == ∂C/∂a⁽ᴸ⁾ . ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ == ′C(a⁽ᴸ⁾, y) . ′σ(z⁽ᴸ⁾) |=> y = expected result
        let mut delta = self.cost_derivative(&last_result.a, &training_data.expected)
            * self.sigma_derivative(&last_result.z);

        // ∇aC |=> Gradient to maximize the network
        let mut nabla = Vec::<LayerNabla>::with_capacity(self.layers.len());
        for (layer, result) in self.layers.iter().zip(network_result).rev() {
            nabla.push(LayerNabla {
                weights: delta // ∂C/∂w⁽ᴸ⁾ == ∂z⁽ᴸ⁾/∂w⁽ᴸ⁾ . 𝛿⁽ᴸ⁾ == a⁽ᴸ⁻¹⁾ᵀ . 𝛿⁽ᴸ⁾
                    .clone()
                    .into_shape((layer.weights.nrows(), 1))
                    .unwrap()
                    .dot(&result.a.into_shape((1, layer.weights.ncols())).unwrap()),
                bias: delta.clone(), // ∂C/∂b⁽ᴸ⁾ == ∂z⁽ᴸ⁾/∂b⁽ᴸ⁾ . 𝛿⁽ᴸ⁾ == 1 . 𝛿⁽ᴸ⁾
            });
            // 𝛿⁽ᴸ⁻¹⁾ == ∂z⁽ᴸ⁾/∂a⁽ᴸ⁻¹⁾ . ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ . 𝛿⁽ᴸ⁾ == w⁽ᴸ⁾ᵀ . 𝛿⁽ᴸ⁾ . ′σ(z⁽ᴸ⁾)
            delta = layer.weights.t().dot(&delta) * self.sigma_derivative(&result.z);
        }
        nabla
    }

    fn sigma(&self, z: f64) -> f64 {
        1.0 / (1.0 + E.powf(-z))
    }

    fn sigma_derivative(&self, zs: &Array1<f64>) -> Array1<f64> {
        zs.map(|z| self.sigma(*z) * (1.0 - self.sigma(*z)))
    }

    fn cost(&self, result: &Array1<f64>, expected: &Array1<f64>) -> f64 {
        let mut cost = 0.0;
        for (r, e) in result.iter().zip(expected.iter()) {
            cost += (r - e).powi(2);
        }
        cost / result.len() as f64
    }

    // Considering the cost function to be: C = ½(result - expected)²
    fn cost_derivative(&self, result: &Array1<f64>, expected: &Array1<f64>) -> Array1<f64> {
        result - expected
    }

    fn to_file(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;

        file.write(&(self.layers.len() as u64).to_be_bytes())?;
        for layer in self.layers.iter() {
            layer.to_file(&mut file)?;
        }
        Ok(())
    }

    fn from_file(filename: &str) -> std::io::Result<Self> {
        let mut file = File::open(filename)?;
        let mut buf = [0 as u8; 8];

        file.read(&mut buf)?;
        let num_layers = u64::from_be_bytes(buf) as usize;
        let mut instance = Self {
            layers: Vec::with_capacity(num_layers),
        };

        for _ in 0..num_layers {
            instance.layers.push(HiddenLayer::from_file(&mut file)?);
        }

        Ok(instance)
    }

    fn to_image(&self, epoch: usize) {
        for (li, layer) in self.layers.iter().enumerate() {
            let file_name = &format!("network_e-{}_l-{}.png", epoch, li);
            let (nrows, ncols) = (layer.weights.nrows() + 1, layer.weights.ncols());

            let size = nrows.max(ncols);
            let layer_drawing_area =
                BitMapBackend::new(&file_name, (size as u32, size as u32)).into_drawing_area();
            let sub_areas = layer_drawing_area.split_evenly((size, size));

            for row in 0..nrows - 1 {
                for column in 0..ncols {
                    let weight = layer.weights[(row, column)];
                    let color = if weight > 0. {
                        RGBColor(0, (weight * 255.) as u8, 0)
                    } else {
                        RGBColor((weight * -255.) as u8, 0, 0)
                    };
                    sub_areas[row * ncols + column].fill(&color).unwrap();
                }
            }
            for column in 0..layer.bias.len() {
                let bias = layer.bias[column];
                let color = if bias > 0. {
                    RGBColor((bias * 255.) as u8, 0, (bias * 255.) as u8)
                } else {
                    RGBColor(0, 0, (bias * -255.) as u8)
                };
                sub_areas[(nrows - 1) * ncols + column]
                    .fill(&color)
                    .unwrap();
            }
        }
    }
}

impl HiddenLayer {
    fn new(num_inputs: usize, num_nodes: usize) -> Self {
        Self {
            weights: Array::random((num_nodes, num_inputs), Uniform::new(-1., 1.)),
            bias: Array::random(num_nodes, Uniform::new(-1., 1.)),
        }
    }

    fn apply(&self, input: &Array1<f64>) -> Array1<f64> {
        &self.weights.dot(input) + &self.bias
    }

    fn to_file(&self, file: &mut File) -> std::io::Result<()> {
        file.write(&self.weights.nrows().to_be_bytes())?;
        file.write(&self.weights.ncols().to_be_bytes())?;
        file.write(&self.bias.len().to_be_bytes())?;

        for w in self.weights.iter() {
            file.write(&w.to_be_bytes())?;
        }
        for b in self.bias.iter() {
            file.write(&b.to_be_bytes())?;
        }

        Ok(())
    }

    fn from_file(file: &mut File) -> std::io::Result<Self> {
        let mut buf = [0 as u8; 8];

        file.read(&mut buf)?;
        let weights_nrows = u64::from_be_bytes(buf) as usize;
        file.read(&mut buf)?;
        let weights_ncols = u64::from_be_bytes(buf) as usize;
        file.read(&mut buf)?;
        let bias_len = u64::from_be_bytes(buf) as usize;

        let mut instance = Self {
            weights: Array::zeros((weights_nrows, weights_ncols)),
            bias: Array::zeros(bias_len),
        };

        for w in instance.weights.iter_mut() {
            file.read(&mut buf)?;
            *w += f64::from_be_bytes(buf);
        }
        for b in instance.bias.iter_mut() {
            file.read(&mut buf)?;
            *b += f64::from_be_bytes(buf);
        }

        Ok(instance)
    }
}

fn main() {
    let (num_rows, num_columns, mut images_pixels) = get_training_images();
    let mut images_labels = get_training_labels();

    let mut training_images = Vec::new();
    while let (Some(pixels), Some(label)) = (images_pixels.pop(), images_labels.pop()) {
        training_images.push(TrainingData {
            input: Array::from(pixels),
            expected: label_to_output_array(label as usize, 10),
        });
    }

    let mut ff_nn = FeedForward::from_file("neural-network.txt")
        .unwrap_or_else(|_| FeedForward::new(num_rows * num_columns, 16, 2, 10));

    let mut epoch = 0;
    loop {
        println!("EPOCH: {}", epoch);

        // ff_nn.to_image(epoch);
        ff_nn.train(&mut training_images, 0.00001);
        ff_nn.to_file("neural-network.txt").unwrap();

        epoch += 1;
    }
}

fn get_training_images() -> (usize, usize, Vec<Vec<f64>>) {
    let training_images_file =
        fs::read("/home/rondao/train-images-idx3-ubyte").expect("Could not read images file.");

    assert_eq!(
        u32::from_be_bytes(training_images_file[..4].try_into().unwrap()),
        2051
    );
    let _num_images = u32::from_be_bytes(training_images_file[4..8].try_into().unwrap());

    let num_rows = u32::from_be_bytes(training_images_file[8..12].try_into().unwrap()) as usize;
    let num_columns = u32::from_be_bytes(training_images_file[12..16].try_into().unwrap()) as usize;

    let image_size = (num_rows * num_columns) as usize;
    let images_pixels = training_images_file[16..]
        .chunks(image_size)
        .map(|pixels| pixels.iter().map(|p| *p as f64 / 255.0).collect())
        .collect();

    (num_rows, num_columns, images_pixels)
}

fn get_training_labels() -> Vec<u8> {
    let training_labels_file =
        fs::read("/home/rondao/train-labels-idx1-ubyte").expect("Could not read images file.");

    assert_eq!(
        u32::from_be_bytes(training_labels_file[..4].try_into().unwrap()),
        2049
    );
    let _num_images = u32::from_be_bytes(training_labels_file[4..8].try_into().unwrap());

    training_labels_file[8..].to_vec()
}

fn label_to_output_array(label: usize, size: usize) -> Array1<f64> {
    Array::from_iter((0..size).map(|value| if value == label { 1.0 } else { 0.0 }))
}

#[allow(dead_code)]
fn image_to_file(file_name: &str, image: &TrainingData, num_rows: usize, num_columns: usize) {
    let root_drawing_area =
        BitMapBackend::new(&file_name, (num_rows as u32, num_columns as u32)).into_drawing_area();
    let sub_areas = root_drawing_area.split_evenly((num_rows, num_columns));

    for row in 0..num_rows {
        for column in 0..num_columns {
            let color = image.input[row * num_rows + column];
            sub_areas[row * num_columns + column]
                .fill(&HSLColor(color, color, color))
                .unwrap();
        }
    }
}
