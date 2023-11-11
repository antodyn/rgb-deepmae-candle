use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};

#[derive(Debug)]
pub struct PatchEmbed {
    proj: Conv2d,
    patch_size: (usize, usize),
    num_patches: usize,
}

impl PatchEmbed {
    pub fn new(
        vb: candle_nn::VarBuilder,
        input_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        let proj = candle_nn::conv2d(
            in_chans,
            embed_dim,
            patch_size,
            Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;
        let num_patches = (input_size / patch_size) * (input_size / patch_size);
        Ok(Self {
            proj,
            patch_size: (patch_size, patch_size),
            num_patches,
        })
    }
    pub fn num_patches(&self) -> usize {
        self.num_patches
    }
}

impl candle_nn::Module for PatchEmbed {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            candle_core::bail!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            candle_core::bail!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;

        // flatten embeddings.
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}
