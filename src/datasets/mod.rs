mod rgb_tvt;
use clap::ValueEnum;
pub use rgb_tvt::{RGBDataloader, RGBDataset};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum WhichDataset {
    Original,
    LEnhanced,
    BEnhanced,
    LbEnhanced,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Split {
    Train,
    Val,
    Test,
}

fn get_dataset_path(dataset: WhichDataset, split: Split) -> String {
    let p = match dataset {
        WhichDataset::Original => String::from("/home/infj-t/Space/Walnut/Datasets/Now/O"),
        WhichDataset::LEnhanced => String::from("/home/infj-t/Space/Walnut/Datasets/Now/L"),
        WhichDataset::BEnhanced => String::from("/home/infj-t/Space/Walnut/Datasets/Now/b"),
        WhichDataset::LbEnhanced => String::from("/home/infj-t/Space/Walnut/Datasets/Now/Lb"),
    };
    log::info!("Load dataset: {:?}", dataset);
    p + match split {
        Split::Train => "/Train",
        Split::Val => "/Val",
        Split::Test => "/Test",
    }
}

pub fn get_dataset(dataset: WhichDataset, split: Split, seed: usize) -> RGBDataset {
    let full_path = get_dataset_path(dataset, split);
    let shuffle = matches!(split, Split::Train);
    RGBDataset::new(&full_path, seed, shuffle)
}
