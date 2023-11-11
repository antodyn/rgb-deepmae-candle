use crate::core::Learner;
use crate::datasets::{get_dataset, Split};
use crate::models::{get_model, ModelTrait};
use candle_core::{DType, Device, Result, D};
use candle_nn::{Optimizer, SGD};
use std::time::Instant;

pub fn count_size(vm: &candle_nn::VarMap) -> usize {
    vm.all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum()
}
pub fn count_size_human(vm: &candle_nn::VarMap) -> f32 {
    count_size(vm) as f32 / 32. / 8. / 2f32.powf(10.0)
}
fn save_safetensors(lnr: &Learner, vm: &candle_nn::VarMap, is_max: bool) {
    // max == true, save max accuracy varmap
    // max == false, save latest varmap
    let save_path = std::path::Path::new(&lnr.recoder_home)
        .join(&lnr.name)
        .join(if is_max {
            "max_accuracy.safetensors"
        } else {
            "latest_epoch.safetensors"
        });
    match vm.save(&save_path) {
        Err(why) => log::error!("Failed to save varmap to safetensors, because of {}", why),
        Ok(_) => log::info!("Saved weights to {:?}", save_path.display()),
    }
}

const MIN_LEARNING_RATE: f64 = 0.00001;

pub fn run(lnr: &Learner) -> anyhow::Result<()> {
    assert!(lnr.start_epoch <= lnr.epochs);

    let device = Device::cuda_if_available(0)?;
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = get_model(vb, lnr.model);
    let mut opt = SGD::new(varmap.all_vars(), lnr.learning_rate).unwrap();
    let mut step_lr = lnr.learning_rate;
    log::info!("learning_rate init: {}", lnr.learning_rate);
    log::info!("varmap size: {:?} MB", count_size_human(&varmap));
    log::info!("batch size: {}", lnr.batch_size);
    // let adamw_params = candle_nn::ParamsAdamW {
    //     lr: lnr.learning_rate,
    //     ..Default::default()
    // };
    // let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    if lnr.load {
        let save_path = std::path::PathBuf::from(&lnr.recoder_home)
            .join(&lnr.name)
            .join("latest_epoch.safetensors");
        log::info!("loading weights from {:?}", save_path.display());
        varmap.load(&save_path)?
    }
    let mut max_acc: f32 = 0.;
    let mut max_acc_epoch: u32 = 0;
    for epoch in lnr.start_epoch..=lnr.epochs {
        let mut loop_time = std::time::Instant::now();
        let avg_loss = train(&model, &mut opt, lnr, device.clone(), epoch as usize)?;
        log::info!(
            "Epoch {epoch:3} -> train average loss: {avg_loss:?}, train time: {:5.2?}",
            loop_time.elapsed()
        );

        loop_time = Instant::now();
        let val_acc = valid(&model, lnr, device.clone())?;
        if max_acc <= val_acc {
            max_acc = val_acc;
            max_acc_epoch = epoch;
            if lnr.save {
                save_safetensors(lnr, &varmap, true);
            }
        }
        log::info!(
            "Epoch {epoch:3} -> validation accuracy: {:5.4}%, valid time: {:5.2?}, max accuracy is {:5.4} in epoch {:3}",
            val_acc * 100.0,
            loop_time.elapsed(),
            max_acc * 100.0, 
            max_acc_epoch,
        );

        if epoch % 10 == 0 {
            save_safetensors(lnr, &varmap, false);
        }

        if epoch % 30 == 0 {
            if step_lr < MIN_LEARNING_RATE {
                step_lr = MIN_LEARNING_RATE;
            } else {
                step_lr *= 0.1;
            }
            opt.set_learning_rate(step_lr);
            log::info!("learning_rate step after epoch {} : {}", epoch, step_lr);
        }
    }
    save_safetensors(lnr, &varmap, false);

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
    model: &Box<dyn ModelTrait>,
    opt: &mut SGD,
    lnr: &Learner,
    device: Device,
    epoch: usize,
) -> Result<f32> {
    let dataset = get_dataset(lnr.dataset, Split::Train, epoch);
    let mut sum_loss = 0f32;
    for (i, (images, labels)) in dataset
        .batcher(lnr.batch_size as usize, device.clone())
        .enumerate()
    {
        let images = images.to_device(&device)?;
        let (logits, loss) = model.forward(&images)?; // loss is option
        let log_sm = (candle_nn::ops::log_softmax(&logits, D::Minus1)?
            .broadcast_add(&loss.unwrap().unsqueeze(D::Minus1)?))?;
        let loss = candle_nn::loss::cross_entropy(&log_sm, &labels)?;
        opt.backward_step(&loss)?;
        let loss_scalar = loss.to_vec0::<f32>()?;
        log::debug!("[Epoch: {epoch:?};{i:?}], batch loss: {:?}", loss_scalar);
        sum_loss += loss_scalar;
    }
    Ok(sum_loss / dataset.len() as f32)
}

fn valid(model: &Box<dyn ModelTrait>, lnr: &Learner, device: Device) -> Result<f32> {
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

fn test(model: &Box<dyn ModelTrait>, lnr: &Learner, device: Device) -> Result<f32> {
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
