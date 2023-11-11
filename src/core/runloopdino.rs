use crate::core::Learner;
use crate::datasets::{get_dataset, Split};
use crate::models::DinoVisionTransformer;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Module, Optimizer, SGD};
use std::sync::mpsc;
use std::thread;
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
fn count_tensor_size_human(t: &Tensor) -> f32 {
    t.elem_count() as f32 / 32. / 8.
}
pub fn rund(lnr: &Learner) -> anyhow::Result<()> {
    assert!(lnr.start_epoch <= lnr.epochs);

    let device = Device::cuda_if_available(0)?;
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = DinoVisionTransformer::new(vb, 12, 384, 6)?;
    let mut opt = SGD::new(varmap.all_vars(), lnr.learning_rate).unwrap();
    println!("varmap size: {:?} MB", count_size_human(&varmap));
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
    for epoch in lnr.start_epoch..=lnr.epochs {
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
    model: &DinoVisionTransformer,
    opt: &mut SGD,
    lnr: &Learner,
    device: Device,
    epoch: usize,
) -> Result<f32> {
    let dataset = get_dataset(lnr.dataset, Split::Train, epoch);
    println!("Train dataset len: {:?}", dataset.len());
    let mut sum_loss = 0f32;
    let chunk: Vec<Vec<std::path::PathBuf>> = dataset
        .images
        .chunks(lnr.batch_size as usize)
        .map(|npz| npz.to_vec())
        .collect::<Vec<_>>();
    let len_chunk = chunk.len();
    assert!(len_chunk > 1);

    for (i, batch) in dataset.images.chunks(lnr.batch_size as usize).enumerate() {
        let (images, labels) = to_batcher(batch.to_vec());
        let images = images.to_device(&device)?;
        let labels = labels.to_device(&device)?;
        // for (i, (images, labels)) in dataset.batcher(lnr.batch_size as usize, device).enumerate() {
        // println!(
        //     "tensor size -> images: {:?} MB, labels: {:?} MB.",
        //     count_tensor_size_human(&images),
        //     count_tensor_size_human(&labels),
        // );
        let logits = model.forward(&images)?;
        let log_sm = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let loss = candle_nn::loss::cross_entropy(&log_sm, &labels)?;
        opt.backward_step(&loss)?;
        let loss_scalar = loss.to_vec0::<f32>()?;
        log::debug!("[Epoch: {epoch:?};{i:?}], batch loss: {:?}", loss_scalar);
        sum_loss += loss_scalar;
    }
    Ok(sum_loss / dataset.len() as f32)
}

fn valid(model: &DinoVisionTransformer, lnr: &Learner, device: Device) -> Result<f32> {
    let val_dataset = get_dataset(lnr.dataset, Split::Val, lnr.seed);
    let mut val_sum_ok = 0f32;
    for (images, labels) in val_dataset.batcher(lnr.batch_size as usize, device) {
        let val_logits = model.forward(&images).unwrap();
        val_sum_ok += val_logits
            .argmax(D::Minus1)?
            .eq(&labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_vec0::<f32>()?;
    }
    Ok(val_sum_ok / val_dataset.len() as f32)
}

fn test(model: &DinoVisionTransformer, lnr: &Learner, device: Device) -> Result<f32> {
    let test_dataset = get_dataset(lnr.dataset, Split::Test, lnr.seed);
    let mut test_sum_ok = 0f32;
    for (images, labels) in test_dataset.batcher(lnr.batch_size as usize, device) {
        let val_logits = model.forward(&images).unwrap();
        test_sum_ok += val_logits
            .argmax(D::Minus1)?
            .eq(&labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_vec0::<f32>()?;
    }
    Ok(test_sum_ok / test_dataset.len() as f32)
}

pub fn to_batcher(batcher: Vec<std::path::PathBuf>) -> (Tensor, Tensor) {
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        for img in batcher {
            let temp_np: Vec<(String, Tensor)> = Tensor::read_npz(img).unwrap();
            let image = &temp_np[0];
            let label = &temp_np[1];
            let _ = tx.send((image.1.clone(), label.1.clone().unsqueeze(0).unwrap()));
        }
    });
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for received in rx {
        images.push(received.0);
        labels.push(received.1);
    }
    handle.join().unwrap();
    //
    // let (images, labels) = batcher
    //     .par_iter()
    //     .map(|p| {
    //         let temp_np: Vec<(String, Tensor)> = Tensor::read_npz(p).unwrap();
    //         let image = &temp_np[0];
    //         let label = &temp_np[1];
    //         (image.1.clone(), label.1.clone().unsqueeze(0).unwrap())
    //     })
    //     .collect::<(Vec<Tensor>, Vec<Tensor>)>();
    let images = Tensor::cat(&images, 0).unwrap();
    let labels = Tensor::cat(&labels, 0).unwrap();
    (images, labels)
}
