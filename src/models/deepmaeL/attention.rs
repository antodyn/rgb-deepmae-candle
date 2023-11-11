use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, Linear, VarBuilder};

#[derive(Debug)]
pub struct Attention {
    re_atten: Option<Conv2d>,
    var_norm: Option<BatchNorm>,
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
}

impl Attention {
    pub fn new(vb: VarBuilder, in_dim: usize, num_heads: usize, re_atten: bool) -> Result<Self> {
        let (re_atten, var_norm) = if re_atten {
            (
                Some(candle_nn::conv2d(
                    num_heads,
                    num_heads,
                    1,
                    Conv2dConfig {
                        stride: 1,
                        ..Default::default()
                    },
                    vb.pp("re_atten"),
                )?),
                Some(candle_nn::batch_norm(num_heads, 1e-5, vb.pp("var_norm"))?),
            )
        } else {
            (None, None)
        };

        let qkv = candle_nn::linear(in_dim, in_dim * 3, vb.pp("qkv"))?;
        let proj = candle_nn::linear(in_dim, in_dim, vb.pp("proj"))?;
        let scale = 1. / ((in_dim / num_heads) as f64).sqrt();

        Ok(Self {
            re_atten,
            var_norm,
            qkv,
            proj,
            num_heads,
            scale,
        })
    }
}

impl candle_nn::Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)? // 02134
            .transpose(0, 1)? // 20134
            .transpose(2, 3)?; // 20314
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        // re_attention
        let attn = if self.re_atten.is_some() && self.var_norm.is_some() {
            (self.var_norm.clone().unwrap().forward(
                &self
                    .re_atten
                    .clone()
                    .unwrap()
                    .forward(&candle_nn::ops::softmax(&q.matmul(&k.t()?)?, D::Minus1)?)?,
            )? * self.scale)?
        } else {
            candle_nn::ops::softmax(&q.matmul(&k.t()?)?, D::Minus1)?
        };
        let attn = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&attn)
    }
}
