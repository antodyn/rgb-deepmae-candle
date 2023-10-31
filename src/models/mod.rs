mod deepmae;
pub use deepmae::DeepMAE;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum WhichModel {
    DeepMAEC16,
    DeepMAEC23,
    DeepMAEC28,
    DeepMAEC33,
}

pub fn get_model(
    vb: candle_nn::var_builder::VarBuilderArgs<'_, Box<dyn candle_nn::var_builder::SimpleBackend>>,
    dev: candle_core::Device,
    model: WhichModel,
) -> DeepMAE {
    match model {
        WhichModel::DeepMAEC16 => deepmae::DeepMAE::new(
            vb,
            dev,
            64,
            4,
            12 * 16,
            12,
            16,
            0.5,
            16 * 16,
            16,
            16,
            false,
            1,
        )
        .unwrap(),
        WhichModel::DeepMAEC23 => deepmae::DeepMAE::new(
            vb,
            dev,
            64,
            4,
            12 * 23,
            12,
            16,
            0.5,
            16 * 23,
            16,
            16,
            false,
            1,
        )
        .unwrap(),
        WhichModel::DeepMAEC28 => deepmae::DeepMAE::new(
            vb,
            dev,
            64,
            4,
            12 * 28,
            12,
            16,
            0.5,
            16 * 28,
            16,
            16,
            false,
            1,
        )
        .unwrap(),
        WhichModel::DeepMAEC33 => deepmae::DeepMAE::new(
            vb,
            dev,
            64,
            4,
            12 * 33,
            12,
            16,
            0.5,
            16 * 33,
            16,
            16,
            false,
            1,
        )
        .unwrap(),
    }
}
