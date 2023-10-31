mod parser;
mod recoder;
mod runloops;
pub use parser::Learner;
pub use recoder::build_log_config;
pub use runloops::run;

pub trait ModelTrait {}
pub trait DatasetTrait {}
pub trait OptimTrait {}
