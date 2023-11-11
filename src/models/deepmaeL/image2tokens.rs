use candle_core::{Result, Tensor};
use candle_nn::{BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, VarBuilder};

#[derive(Debug)]
pub struct Image2Tokens {
    conv: Conv2d,
    bn: BatchNorm,
}
impl Image2Tokens {
    pub fn new(
        vb: candle_nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> candle_core::Result<Self> {
        let conv = candle_nn::conv2d_no_bias(
            in_channels,
            out_channels,
            kernel_size,
            Conv2dConfig {
                stride,
                padding,
                ..Default::default()
            },
            vb.pp("conv2d"),
        )?;
        let bn = candle_nn::batch_norm(
            out_channels,
            BatchNormConfig {
                eps: 1e-5,
                affine: true,
                remove_mean: false,
            },
            vb.pp("bn"),
        )?;
        Ok(Self { conv, bn })
    }
}
impl candle_nn::Module for Image2Tokens {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.conv)?.apply(&self.bn)?.max_pool2d(2)
    }
}
// i2t builder, in_channels=3, out_channels=64, kernel_size=7, out_width=input_width/4
pub fn i2t_64_256(vb: VarBuilder) -> Result<Image2Tokens> {
    Image2Tokens::new(vb, 3, 64, 7, 2, 3)
}
