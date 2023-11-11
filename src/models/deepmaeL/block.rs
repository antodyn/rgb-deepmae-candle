use super::{Attention, Mlp};
use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};

#[derive(Debug)]
pub struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    pub fn new(
        vb: VarBuilder,
        embed_dim: usize,
        num_heads: usize,
        re_attention: bool,
    ) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("norm1"))?;
        let attn = Attention::new(vb.pp("attn"), embed_dim, num_heads, re_attention)?;
        let norm2 = candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), embed_dim, Some(embed_dim * 4))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }
}

impl candle_nn::Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.norm1)?.apply(&self.attn)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = &xs.apply(&self.norm2)?.apply(&self.mlp)?;
        xs + residual
    }
}
