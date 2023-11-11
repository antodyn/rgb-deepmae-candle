mod deepmae;
pub use deepmae::DeepMAE;
mod deepmaeL;
pub use deepmaeL::DeepMAEL;
mod dinov2;
pub use dinov2::DinoVisionTransformer;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum WhichModel {
    DeepMAEC16,
    DeepMAEC23,
    DeepMAEC28,
    DeepMAEC33,
    // DeepMAELC16,
}

pub trait ModelTrait {
    fn forward(
        &self,
        xs: &candle_core::Tensor,
    ) -> candle_core::Result<(candle_core::Tensor, Option<candle_core::Tensor>)>;
}

pub fn get_model(
    vb: candle_nn::var_builder::VarBuilderArgs<'_, Box<dyn candle_nn::var_builder::SimpleBackend>>,
    model: WhichModel,
) -> Box<dyn ModelTrait> {
    log::info!("Model: {:?}", model);
    match model {
        WhichModel::DeepMAEC16 => Box::new(
            deepmae::DeepMAE::new(vb, 64, 4, 12 * 16, 12, 16, 0.5, 16 * 16, 16, 16, false, 1)
                .unwrap(),
        ),
        WhichModel::DeepMAEC23 => Box::new(
            deepmae::DeepMAE::new(vb, 64, 4, 12 * 23, 12, 16, 0.5, 16 * 24, 16, 16, false, 1)
                .unwrap(),
        ),
        WhichModel::DeepMAEC28 => Box::new(
            deepmae::DeepMAE::new(vb, 64, 4, 12 * 28, 12, 16, 0.5, 16 * 28, 16, 16, false, 1)
                .unwrap(),
        ),
        WhichModel::DeepMAEC33 => Box::new(
            deepmae::DeepMAE::new(vb, 64, 4, 12 * 33, 12, 16, 0.5, 16 * 34, 16, 16, false, 1)
                .unwrap(),
        ),
        // WhichModel::DeepMAELC16 => Box::new(
        //     deepmaeL::DeepMAEL::new(vb, 64, 4, 12 * 16, 12, 16, 0.5, 16 * 16, 16, 16, false, 1)
        //         .unwrap(),
        // ),
    }
}
