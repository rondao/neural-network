use plotters::prelude::*;
use std::{convert::TryInto, u32, usize};
use std::fs;

struct TrainingImage {
    pixels: Vec<u8>,
    label: u8,
}

fn main() {
    let (num_rows, num_columns, mut images_pixels) = get_training_images();
    let mut images_labels = get_training_labels();

    let mut training_images:Vec<TrainingImage> = Vec::new();
    while let (Some(pixels),Some(label)) = (images_pixels.pop(), images_labels.pop()) {
        training_images.push(TrainingImage{pixels, label});
    }


    for (idx, training_image) in training_images.iter().enumerate() {
        let file_name = &format!("plotted-{}.png", idx);
        println!("{} - {}", file_name, training_image.label);
        let root_drawing_area =
            BitMapBackend::new(file_name, (num_rows as u32, num_columns as u32))
                .into_drawing_area();
        let sub_areas = root_drawing_area.split_evenly((num_rows, num_columns));

        for row in 0..num_rows {
            for column in 0..num_columns {
                let color = training_image.pixels[row * num_rows + column];
                sub_areas[row * num_rows + column]
                    .fill(&RGBColor(color, color, color))
                    .unwrap();
            }
        }
    }
}

fn get_training_images() -> (usize, usize, Vec<Vec<u8>>) {
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
    let images_pixels: Vec<Vec<u8>> = training_images_file[16..]
        .chunks(image_size)
        .map(|x| x.to_vec())
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

    return training_labels_file[8..].to_vec();
}
