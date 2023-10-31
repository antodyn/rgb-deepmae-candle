use candle_core::{Result, Tensor};
use candle_nn::{Linear, VarBuilder};

#[derive(Debug)]
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn new(vb: VarBuilder, in_dim: usize, hidden_dim: Option<usize>) -> Result<Self> {
        let hidden_dim = if let Some(hidden_dim) = hidden_dim {
            hidden_dim
        } else {
            in_dim * 3
        };
        let fc1 = candle_nn::linear(in_dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_dim, in_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl candle_nn::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs.apply(&self.fc1)?.gelu()?.apply(&self.fc2)
        xs.apply(&self.fc1)?.relu()?.apply(&self.fc2)
    }
}
