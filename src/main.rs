use plotters::prelude::*;
use std::convert::TryInto;
use std::fs;

fn main() {
    let training_images_file =
        fs::read("/home/rondao/train-images-idx3-ubyte").expect("Could not read images file.");

    let magic_number = u32::from_be_bytes(training_images_file[..4].try_into().unwrap());
    let number_of_images = u32::from_be_bytes(training_images_file[4..8].try_into().unwrap());
    let number_of_rows =
        u32::from_be_bytes(training_images_file[8..12].try_into().unwrap()) as usize;
    let number_of_columns =
        u32::from_be_bytes(training_images_file[12..16].try_into().unwrap()) as usize;

    println!(
        "[{}] {} ({},{})",
        magic_number, number_of_images, number_of_rows, number_of_columns
    );

    let image_size = number_of_rows * number_of_columns;
    let training_images = &training_images_file[16..];

    for (idx, image_bytes) in training_images.chunks(image_size).enumerate() {
        let file_name = &format!("plotted-{}.png", idx);
        let root_drawing_area =
            BitMapBackend::new(file_name, (number_of_rows as u32, number_of_columns as u32))
                .into_drawing_area();
        let sub_areas = root_drawing_area.split_evenly((number_of_rows, number_of_columns));

        for row in 0..number_of_rows {
            for column in 0..number_of_columns {
                let color = image_bytes[row * number_of_rows + column];
                sub_areas[row * number_of_rows + column]
                    .fill(&RGBColor(color, color, color))
                    .unwrap();
            }
        }
    }
}
