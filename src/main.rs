extern crate nalgebra as na;

use na::{DMatrix, DVector};
use plotters::prelude::*;
use rand::Rng;
use std::{borrow::Borrow, convert::TryInto, fs, u32, usize};

struct TrainingImage {
    pixels: DVector<f64>,
    label: u8,
}

struct HiddenLayer {
    weights: DMatrix<f64>,
    bias: DVector<f64>,
}

impl HiddenLayer {
    fn new(num_inputs: usize, num_nodes: usize) -> Self {
        Self {
            weights: DMatrix::from_vec(num_nodes, num_inputs, (0..num_inputs*num_nodes).map(|_| rand::thread_rng().gen_range(-10.0..10.0)).collect()),
            bias: DVector::from_vec((0..num_nodes).map(|_| rand::thread_rng().gen_range(-10.0..10.0)).collect()),
        }
    }

    fn apply(&self, input: &DVector<f64>) -> DVector<f64> {
        let result = self.weights.borrow() * input + self.bias.borrow();
        result.map(|value| f64::max(0.0, value))
    }
}
struct FeedForward {
    layers: Vec<HiddenLayer>,
}

impl FeedForward {
    fn new(num_inputs: usize, num_nodes: usize, num_layers: usize, num_outputs: usize) -> Self {
        let mut instance = Self {
            layers: Vec::with_capacity(num_layers),
        };

        instance.layers.push(HiddenLayer::new(num_inputs, num_nodes));
        for _ in 1..num_layers {
            instance.layers.push(HiddenLayer::new(num_nodes, num_nodes));
        }
        instance.layers.push(HiddenLayer::new(num_nodes, num_outputs));

        instance
    }

    fn apply(&self, input: &DVector<f64>) -> DVector<f64> {
        let mut result = self.layers[0].apply(input);
        for layer in self.layers[1..].iter() {
            result = layer.apply(&result);
        }
        result
    }
}

fn main() {
    let (num_rows, num_columns, mut images_pixels) = get_training_images();
    let mut images_labels = get_training_labels();

    let mut training_images = Vec::new();
    while let (Some(pixels),Some(label)) = (images_pixels.pop(), images_labels.pop()) {
        training_images.push(TrainingImage{pixels: DVector::from(pixels), label});
    }

    let ff_nn = FeedForward::new(num_rows * num_columns, 16, 2, 10);

    let mut images_result = Vec::new();
    for image in training_images.iter() {
        images_result.push(ff_nn.apply(image.pixels.borrow()));
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

#[allow(dead_code)]
fn image_to_file(file_name: String, image: TrainingImage, num_rows: usize, num_columns: usize) {
    let root_drawing_area =
        BitMapBackend::new(&file_name, (num_rows as u32, num_columns as u32))
            .into_drawing_area();
    let sub_areas = root_drawing_area.split_evenly((num_rows, num_columns));

    for row in 0..num_rows {
        for column in 0..num_columns {
            let color = image.pixels[row * num_rows + column];
            sub_areas[row * num_rows + column]
                .fill(&HSLColor(color, color, color))
                .unwrap();
        }
    }
}