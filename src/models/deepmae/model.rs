use super::{i2t_64_256, tensor_ops, Block, Image2Tokens, PatchEmbed};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

pub struct DeepMAE {
    device: Device,
    i2t: Image2Tokens,
    patch_embed: PatchEmbed,
    encoder_blocks: Vec<Block>,
    cls_token: Tensor,
    pos_embed: Tensor,
    norm: LayerNorm,
    mask_ratio: f32,
    decoder_embed: Linear,
    mask_token: Tensor,
    decoder_pos_embed: Tensor,
    decoder_blocks: Vec<Block>,
    decoder_norm: LayerNorm,
    decoder_pred: Linear,
    norm_pix_loss: bool,
    classify_embed: Linear,
    classify_pos_embed: Tensor,
    classify_blocks: Vec<Block>,
    classify_norm: LayerNorm,
    head: Linear,
}
const NUM_CLASSES: usize = 4;
impl DeepMAE {
    pub fn new(
        vb: VarBuilder,
        device: Device,
        i2t_c: usize,
        patch_size: usize,
        embed_dim: usize,
        num_heads: usize,
        depth: usize,
        mask_ratio: f32,
        decoder_embed_dim: usize,
        decoder_heads: usize,
        decoder_depth: usize,
        norm_pix_loss: bool,
        classify_depth: usize,
    ) -> Result<Self> {
        let i2t = i2t_64_256(vb.pp("i2t")).unwrap();
        // 64 is i2t out_w
        let patch_embed =
            PatchEmbed::new(vb.pp("patch_embed"), 64, patch_size, i2t_c, embed_dim).unwrap();
        let cls_token = vb.get((1, 1, embed_dim), "cls_token").unwrap();
        let num_tokens = 1;
        let pos_embed = vb
            .get(
                (1, patch_embed.num_patches() + num_tokens, embed_dim),
                "pos_embed",
            )
            .unwrap();
        let norm = candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("norm")).unwrap();

        // Encoder
        let vb_blocks = vb.pp("encoder_blocks");
        let encoder_blocks = (0..depth)
            .map(|i| Block::new(vb_blocks.pp(&i.to_string()), embed_dim, num_heads, true))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert!((0.0..1.0).contains(&mask_ratio));

        // Decoder
        let decoder_embed =
            candle_nn::linear(embed_dim, decoder_embed_dim, vb.pp("decoder_embed")).unwrap();
        let mask_token = vb.get((1, 1, decoder_embed_dim), "mask_token").unwrap();
        let decoder_pos_embed = vb
            .get(
                (1, patch_embed.num_patches() + num_tokens, decoder_embed_dim),
                "decoder_pos_embed",
            )
            .unwrap();
        let vb_blocks = vb.pp("decoder_blocks");
        let decoder_blocks = (0..decoder_depth)
            .map(|i| {
                Block::new(
                    vb_blocks.pp(&i.to_string()),
                    decoder_embed_dim,
                    decoder_heads,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let decoder_norm =
            candle_nn::layer_norm(decoder_embed_dim, 1e-5, vb.pp("decoder_norm")).unwrap();
        let decoder_pred = candle_nn::linear(
            decoder_embed_dim,
            // patch_size.pow(2) * 3,
            16 * 16 * 3,
            vb.pp("decoder_pred"),
        )
        .unwrap();

        // Classfiy
        let classify_embed =
            // candle_nn::linear(patch_size.pow(2) * 3, embed_dim, vb.pp("classify_embed")).unwrap();
            candle_nn::linear(16 * 16 * 3, embed_dim, vb.pp("classify_embed")).unwrap();
        let classify_pos_embed = vb
            .get(
                (
                    1,
                    ((patch_embed.num_patches() as f32) * mask_ratio) as usize,
                    embed_dim,
                ),
                "classify_pos_embed",
            )
            .unwrap();
        let vb_blocks = vb.pp("classify_blocks");
        let classify_blocks = (0..classify_depth)
            .map(|i| {
                Block::new(
                    vb_blocks.pp(&i.to_string()),
                    // decoder_embed_dim,
                    embed_dim,
                    decoder_heads,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let classify_norm = candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("classify_norm")).unwrap();
        let head = candle_nn::linear(embed_dim, NUM_CLASSES, vb.pp("head")).unwrap();

        Ok(Self {
            device,
            i2t,
            patch_embed,
            encoder_blocks,
            cls_token,
            pos_embed,
            norm,
            mask_ratio,
            decoder_embed,
            mask_token,
            decoder_pos_embed,
            decoder_blocks,
            decoder_norm,
            decoder_pred,
            norm_pix_loss,
            classify_embed,
            classify_pos_embed,
            classify_blocks,
            classify_norm,
            head,
        })
    }
}
// impl Module for DeepMAE {
impl DeepMAE {
    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (latent, mask, ids_restore, ids_nokeep) = self.forward_encoder(xs).unwrap();
        let pred = self.forward_decoder(&latent, &ids_restore).unwrap();
        let loss = self.forward_loss(xs, &pred, &mask).unwrap();
        let classify = self.forward_classify(&pred, &ids_nokeep).unwrap();

        Ok((classify, loss))
    }
}

impl DeepMAE {
    fn interpolate_pos_encoding(&self, xs: &Tensor, w: usize, h: usize) -> Result<Tensor> {
        // no cat cls token
        let npatch = xs.dim(1).unwrap();
        // let npatch = xs.dim(1).unwrap() - 1;
        let n = self.pos_embed.dim(1).unwrap() - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            Ok(xs.clone())
        } else {
            panic!("sqrt_n: {sqrt_n}, n: {n}, npatch: {npatch}, w: {w}, h: {h}");
        }
        // let class_pos_embed = self.pos_embed.i((.., ..1)).unwrap();
        // let patch_pos_embed = self.pos_embed.i((.., 1..)).unwrap();
        // let dim = xs.dim(D::Minus1).unwrap();
        // let PATCH_SIZE = 4;
        // let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        // let patch_pos_embed = patch_pos_embed
        //     .reshape((1, sqrt_n as usize, sqrt_n as usize, dim)).unwrap()
        //     .transpose(2, 3).unwrap()
        //     .transpose(1, 2).unwrap();
        // // This uses bicubic interpolation in the original implementation.
        // let patch_pos_embed = patch_pos_embed.upsample_nearest2d(h0 as usize, w0 as usize).unwrap();
        // let el_count = patch_pos_embed.shape().elem_count();
        // let patch_pos_embed =
        //     patch_pos_embed
        //         .transpose(1, 2).unwrap()
        //         .transpose(2, 3).unwrap()
        //         .reshape((1, el_count / dim, dim)).unwrap();
        // Tensor::cat(&[&class_pos_embed, &patch_pos_embed], 1)
    }

    fn random_masking(
        &self,
        xs: &Tensor,
        mask_ratio: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (n, l, d) = xs.dims3().unwrap();
        let len_keep = (l as f32 * (1. - mask_ratio)) as usize;
        let noise = Tensor::rand(
            0.0f32,
            1.0f32,
            candle_core::Shape::from_dims(&[n, l]),
            &self.device,
        )
        .unwrap();
        // rand noise
        let (ids_shuffle, ids_restore) = tensor_ops::tensor_argsort(&noise, &self.device).unwrap();
        let ids_restore = ids_restore.contiguous().unwrap();
        let ids_keep = ids_shuffle
            .i((.., ..len_keep))
            .unwrap()
            .contiguous()
            .unwrap();
        let ids_nokeep = ids_shuffle
            .i((.., len_keep..))
            .unwrap()
            .contiguous()
            .unwrap();
        // x_maskd
        let gather_index = ids_keep
            .unsqueeze(D::Minus1)
            .unwrap()
            .repeat((1, 1, d))
            .unwrap()
            .contiguous()
            .unwrap();
        let x_maskd = xs.gather(&gather_index, 1).unwrap();
        // mask
        let mask_0 = Tensor::zeros((n, len_keep), DType::F32, &self.device).unwrap();
        let mask_1 = Tensor::ones((n, l - len_keep), DType::F32, &self.device).unwrap();
        let mask = Tensor::cat(&[mask_0, mask_1], 1)
            .unwrap()
            .contiguous()
            .unwrap();
        let mask = mask.gather(&ids_restore, 1).unwrap();

        Ok((x_maskd, mask, ids_restore, ids_nokeep))
    }
    fn forward_encoder(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b, _nc, w, h) = xs.dims4().unwrap();
        let xs = self.i2t.forward(xs).unwrap();
        let xs = self.patch_embed.forward(&xs).unwrap();
        // cat pos_embed
        let xs = (&xs + &self.interpolate_pos_encoding(&xs, w, h).unwrap()).unwrap();
        // before: [batchsize, 256, embed_dim]
        // random_masking
        let (xs, mask, ids_restore, ids_nokeep) =
            self.random_masking(&xs, self.mask_ratio).unwrap();
        // after: [batchsize, 64, embed_dim] 64 == 256 * (1-0.75)
        // append cls token
        let broadcast_shape = candle_core::Shape::from_dims(&[b, 1, xs.dim(2).unwrap()]);
        let mut xs = Tensor::cat(
            &[&self.cls_token.broadcast_as(broadcast_shape).unwrap(), &xs],
            1,
        )
        .unwrap();
        // after append cls: [batchsize, 64, embed_dim]
        // apply transformer blocks
        for blk in self.encoder_blocks.iter() {
            xs = blk.forward(&xs).unwrap();
        }
        let xs = self.norm.forward(&xs).unwrap();
        Ok((xs, mask, ids_restore, ids_nokeep))
    }
    fn forward_decoder(&self, xs: &Tensor, ids_restore: &Tensor) -> Result<Tensor> {
        //embed tokens
        let xs = self.decoder_embed.forward(xs).unwrap();
        // append mask tokens to sequence
        let mask_tokens = self
            .mask_token
            .repeat((
                xs.dim(0).unwrap(),
                (ids_restore.dim(1).unwrap() + 1 - xs.dim(1).unwrap()),
                1,
            ))
            .unwrap();
        let xs_t = Tensor::cat(&[xs.i((.., 1.., ..)).unwrap(), mask_tokens], 1)
            .unwrap()
            .contiguous()
            .unwrap(); // no cls token
        let xs_t = xs_t
            .gather(
                &ids_restore
                    .unsqueeze(D::Minus1)
                    .unwrap()
                    .repeat((1, 1, xs.dim(2).unwrap()))
                    .unwrap()
                    .contiguous()
                    .unwrap(),
                1,
            )
            .unwrap(); // unshuffle
        let xs = Tensor::cat(&[xs.i((.., ..1, ..)).unwrap(), xs_t], 1).unwrap(); // append cls token
                                                                                 //
        let mut xs = xs.broadcast_add(&self.decoder_pos_embed)?;
        // + &self
        //     .decoder_pos_embed
        //     .repeat((&xs.dim(0)?, 1, 1))?
        //     .contiguous()?)
        // .unwrap(); // add pos embed
        // apply Transformer blocks
        for blk in self.decoder_blocks.iter() {
            xs = blk.forward(&xs).unwrap();
        }
        let xs = self.decoder_norm.forward(&xs).unwrap();
        // predictor projection
        let xs = self.decoder_pred.forward(&xs).unwrap();
        // remove cls token
        let xs = xs.i((.., 1.., ..)).unwrap();
        Ok(xs)
    }
    fn patchify(&self, xs: &Tensor) -> Result<Tensor> {
        //  imgs: (N, 3, H, W)
        //  x: (N, L, patch_size**2 *3)
        // p = self.patch_embed.patch_size[0]
        let p = 16;
        // let p = 4;
        assert_eq!(xs.dim(2).unwrap(), xs.dim(3).unwrap());
        assert!(xs.dim(2).unwrap() % p == 0);
        let h = xs.dim(2).unwrap() / p;
        let w = h;
        //  torch.einsum('nchpwq->nhwpqc', x)
        // x = torch.Tensor(1, 3, 16, 4, 16, 4) 在12288个数值中，可能有三个不相等，可以忽略
        // torch.eq(torch.permute(x, (0, 2, 4, 3, 5, 1)), torch.einsum('nchpwq->nhwpqc', x))
        xs.reshape((xs.dim(0).unwrap(), 3, h, p, w, p))
            .unwrap()
            .permute((0, 2, 4, 3, 5, 1))
            .unwrap()
            .reshape((xs.dim(0).unwrap(), h * w, p.pow(2) * 3))
    }
    fn unpatchify(&self, xs: &Tensor) -> Result<Tensor> {
        //  x: (N, L, patch_size**2 *3)
        //  imgs: (N, 3, H, W)
        //  p = self.patch_embed.patch_size[0]
        let p = 16;
        let h = (xs.dim(1).unwrap() as f64).sqrt() as usize;
        let w = h;
        assert_eq!(h * w, xs.dim(1).unwrap());
        //  x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        //  x = torch.einsum('nhwpqc->nchpwq', x)
        //  imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        xs.reshape((xs.dim(0).unwrap(), h, w, p, p, 3))
            .unwrap()
            .permute((0, 5, 1, 3, 2, 4))
            .unwrap()
            .reshape((xs.dim(0).unwrap(), 3, h * p, w * p))
    }

    // fn calc_pix_loss(&self, target: &Tensor) -> Result<(Tensor, Tensor)> {
    //     let mean = target.mean_keepdim(D::Minus1).unwrap();
    //     // let val = target.var(D::Minus1).unwrap();
    //     Ok((mean, val))
    // }
    fn forward_loss(&self, xs: &Tensor, pred: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // imgs: [N, 3, H, W]
        // pred: [N, L, p*p*3]
        // mask: [N, L], 0 is keep, 1 is remove,
        let mut target = self.patchify(xs).unwrap();
        if self.norm_pix_loss {
            //         mean = target.mean(dim=-1, keepdim=True)
            //         var = target.var(dim=-1, keepdim=True)
            //         target = (target - mean) / (var + 1.e-6)**.5
            panic!("can not norm_pix_loss==ture")
        }

        let mut loss = pred.sub(&target).unwrap().powf(2.).unwrap();
        loss = loss.mean(D::Minus1).unwrap();
        loss = (loss.mul(mask).unwrap().sum(D::Minus1).unwrap() / mask.sum(D::Minus1).unwrap())
            .unwrap(); //  mean loss on removed patches

        Ok(loss)
    }

    fn forward_classify(&self, xs: &Tensor, ids_nokeep: &Tensor) -> Result<Tensor> {
        //  embed tokens
        let xs = self.classify_embed.forward(xs).unwrap();
        let (l, _, d) = xs.dims3().unwrap();
        let xs = xs
            .gather(
                &ids_nokeep
                    .unsqueeze(D::Minus1)
                    .unwrap()
                    .repeat((1, 1, d))
                    .unwrap()
                    .contiguous()
                    .unwrap(),
                1,
            )
            .unwrap();

        // let mut xs = Tensor::cat(&[&xs, &self.classify_pos_embed.repeat((l, 1, 1)).unwrap()], 1).unwrap();
        let mut xs = (&xs + &self.classify_pos_embed.repeat((l, 1, 1)).unwrap()).unwrap();
        //  apply Transformer blocks
        for blk in self.classify_blocks.iter() {
            xs = blk.forward(&xs).unwrap();
        }
        let xs = self.classify_norm.forward(&xs).unwrap();
        let xs = xs.i((.., 0)).unwrap().contiguous().unwrap();
        self.head.forward(&xs)
    }
}
