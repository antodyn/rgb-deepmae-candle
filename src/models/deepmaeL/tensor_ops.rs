use super::sort_float;
use candle_core::{Device, Tensor};
pub fn tensor_argsort(tensor: &Tensor, device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
    let (n, l) = tensor.dims2().unwrap();
    let mut shuffle_indices: Vec<u32> = Vec::new();
    let mut restore_indices: Vec<u32> = Vec::new();
    let tensor_v = tensor.to_vec2::<f32>().unwrap();
    for v in tensor_v.iter() {
        let after_sort_float = sort_float::float_argsort(v);
        shuffle_indices.extend(after_sort_float.clone());
        let after_int_argsort = sort_float::int_argsort(&after_sort_float);
        restore_indices.extend(after_int_argsort.clone());
    }
    let shuffle_indices = Tensor::from_vec(shuffle_indices, (n, l), device).unwrap();
    let restore_indices = Tensor::from_vec(restore_indices, (n, l), device).unwrap();
    Ok((shuffle_indices, restore_indices))
}

