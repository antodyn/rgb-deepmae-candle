use crate::core::Learner;
use crate::datasets::{get_dataset, Split};
use crate::models::{get_model, DeepMAE};
use candle_core::{DType, Device, Result, D};
use candle_nn::{Optimizer, SGD};
use std::time::Instant;

pub fn run(lnr: &Learner) -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = get_model(vb, device.clone(), lnr.model);
    let mut opt = SGD::new(varmap.all_vars(), lnr.learning_rate).unwrap();
    // let adamw_params = candle_nn::ParamsAdamW {
    //     lr: lnr.learning_rate,
    //     ..Default::default()
    // };
    // let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    let save_path = std::path::PathBuf::from(&lnr.recoder_home)
        .join(&lnr.name)
        .join("latest_epoch.safetensors");
    if lnr.load {
        log::info!("loading weights from {:?}", save_path.display());
        varmap.load(&save_path)?
    }
    for epoch in 0..lnr.epochs {
        let mut loop_time = std::time::Instant::now();
        let avg_loss = train(&model, &mut opt, lnr, device.clone(), epoch as usize)?;
        log::info!(
            "Epoch {epoch:3} -> train average loss: {avg_loss:?}, train time: {:5.2?}",
            loop_time.elapsed()
        );

        loop_time = Instant::now();
        let val_acc = valid(&model, lnr, device.clone())?;
        log::info!(
            "Epoch {epoch:3} -> validation accuracy: {:5.2}%, valid time: {:5.2?}",
            val_acc * 100.0,
            loop_time.elapsed()
        );
    }

    if lnr.save {
        varmap.save(&save_path)?;
        log::info!("Saved weights to {:?}", save_path.display());
    }

    let test_time = Instant::now();
    let test_acc = test(&model, lnr, device.clone())?;
    log::info!(
        "Test -> test accuracy: {:5.2}%, test time: {:5.2?}",
        test_acc * 100.0,
        test_time.elapsed()
    );

    Ok(())
}

fn train(
    model: &DeepMAE,
    opt: &mut SGD,
    lnr: &Learner,
    device: Device,
    epoch: usize,
) -> Result<f32> {
    let dataset = get_dataset(lnr.dataset, Split::Train, epoch);
    let mut sum_loss = 0f32;
    for (i, (images, labels)) in dataset.batcher(lnr.batch_size as usize, device).enumerate() {
        let (logits, loss) = model.forward(&images)?;
        let log_sm = (candle_nn::ops::log_softmax(&logits, D::Minus1)?
            .broadcast_add(&loss.unsqueeze(D::Minus1)?))?;
        let loss = candle_nn::loss::cross_entropy(&log_sm, &labels)?;
        opt.backward_step(&loss)?;
        let loss_scalar = loss.to_vec0::<f32>()?;
        log::debug!("[Epoch: {epoch:?};{i:?}], batch loss: {:?}", loss_scalar);
        sum_loss += loss_scalar;
    }
    Ok(sum_loss / dataset.len() as f32)
}

fn valid(model: &DeepMAE, lnr: &Learner, device: Device) -> Result<f32> {
    let val_dataset = get_dataset(lnr.dataset, Split::Val, lnr.seed);
    let mut val_sum_ok = 0f32;
    for (images, labels) in val_dataset.batcher(lnr.batch_size as usize, device) {
        let (val_logits, _) = model.forward(&images).unwrap();
        val_sum_ok += val_logits
            .argmax(D::Minus1)?
            .eq(&labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_vec0::<f32>()?;
    }
    Ok(val_sum_ok / val_dataset.len() as f32)
}

fn test(model: &DeepMAE, lnr: &Learner, device: Device) -> Result<f32> {
    let test_dataset = get_dataset(lnr.dataset, Split::Test, lnr.seed);
    let mut test_sum_ok = 0f32;
    for (images, labels) in test_dataset.batcher(lnr.batch_size as usize, device) {
        let (val_logits, _) = model.forward(&images).unwrap();
        test_sum_ok += val_logits
            .argmax(D::Minus1)?
            .eq(&labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_vec0::<f32>()?;
    }
    Ok(test_sum_ok / test_dataset.len() as f32)
}
