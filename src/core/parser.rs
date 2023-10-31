use crate::datasets::WhichDataset;
use crate::models::WhichModel;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "Deep learning train with candle")]
#[command(author = "Antodyn <antodyn@163.com>")]
#[command(version = "0.1")]
#[command(about = "Image classification using candle", long_about = None)]
pub struct Learner {
    /// Name of this train
    #[arg(short, long)]
    pub name: String,

    /// specity WhichDataset
    #[arg(short, long, value_enum, default_value_t=WhichDataset::Original)]
    pub dataset: WhichDataset,

    /// specity WhichModel
    #[arg(short, long, value_enum, default_value_t=WhichModel::DeepMAEC16)]
    pub model: WhichModel,

    /// learning rate
    #[arg(short, long, default_value_t = 0.05)]
    pub learning_rate: f64,

    /// batch size
    #[arg(short, long, value_parser = clap::value_parser!(u32).range(1..), default_value_t = 16)]
    pub batch_size: u32,

    /// epochs
    #[arg(short, long, value_parser = clap::value_parser!(u32).range(1..), default_value_t = 10)]
    pub epochs: u32,

    /// start epochs
    #[arg(long, default_value_t = 1)]
    pub start_epoch: u32,

    /// seed
    #[arg(short, long, default_value_t = 42)]
    pub seed: usize,

    /// load checkpoint
    #[arg(long, default_value_t = false)]
    pub load: bool,

    /// save checkpoint
    #[arg(long, default_value_t = true)]
    pub save: bool,

    /// recoder home path
    #[arg(short, long, default_value_t = String::from("tmp"))]
    pub recoder_home: String,
}
