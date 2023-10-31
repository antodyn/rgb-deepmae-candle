use candle_core::{DType, Device, Tensor};
use rand::{prelude::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const RGB_IMAGE_SIZE: usize = 256;
const RGB_CHANNELS: usize = 3;

#[derive(Default)]
pub struct RGBDataset {
    images: Vec<PathBuf>,
    labels: Vec<usize>,
    names: Vec<String>,
}
impl RGBDataset {
    pub fn new(dataset_path: &str, seed: usize, shuffle: bool) -> Self {
        let path = PathBuf::from(dataset_path);
        let dataset = Self::default();
        let mut dataset = dataset.level2_dataset(path);
        if shuffle {
            dataset = dataset.shuffle_dataset(seed);
        }
        // log::info!("image example: [0: {:?}]", dataset.names()[0]);
        dataset
    }
    fn level2_dataset<P: AsRef<Path>>(&self, root: P) -> Self {
        let path1 = ["A", "B", "C", "D"]
            .iter()
            .map(|c| root.as_ref().to_path_buf().join(c))
            .collect::<Vec<_>>();
        let mut images = Vec::new();
        let mut labels = Vec::new();
        let mut names = Vec::new();
        path1.into_iter().for_each(|p1| {
            let (imgs, lbls, nms) = self.list_dir(p1);
            images.extend(imgs);
            labels.extend(lbls);
            names.extend(nms);
        });
        Self {
            images,
            labels,
            names,
        }
    }
    fn shuffle_dataset(&self, seed: usize) -> Self {
        let mut image_label_name = self
            .images
            .clone()
            .into_iter()
            .zip(self.labels.clone())
            .zip(self.names.clone())
            .map(|((img, lbl), nm)| (img, lbl, nm))
            .collect::<Vec<(_, _, _)>>();
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        log::info!("Seed: {}", seed);
        image_label_name.shuffle(&mut rng);
        let mut images = Vec::new();
        let mut labels = Vec::new();
        let mut names = Vec::new();
        image_label_name.into_iter().for_each(|(img, lbl, nm)| {
            images.push(img);
            labels.push(lbl);
            names.push(nm);
        });
        Self {
            images,
            labels,
            names,
        }
    }
    pub fn images(&self) -> &[PathBuf] {
        &self.images
    }
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }
    pub fn names(&self) -> &[String] {
        &self.names
    }
    pub fn len(&self) -> usize {
        assert_eq!(self.images.len(), self.labels.len());
        self.images.len()
    }
    pub fn batcher(&self, batch_size: usize, device: Device) -> RGBDataloader {
        RGBDataloader::new(
            self.images
                .clone()
                .into_iter()
                .zip(self.labels.clone())
                .collect::<Vec<(_, _)>>()
                .chunks(batch_size)
                .map(|imglbls| imglbls.to_vec())
                .collect(),
            device,
        )
    }
    fn get_label_name(&self, path: &str) -> (usize, String) {
        let path_vec = path.split('/').collect::<Vec<_>>();
        assert!(path_vec.len() > 1); // At least, like `Class/1.jpg`
        let mut path_iter = path_vec.iter().rev();
        let name: String = path_iter.next().unwrap().to_string();
        let label: usize = match path_iter.next() {
            Some(&"A") => 0,
            Some(&"B") => 1,
            Some(&"C") => 2,
            Some(&"D") => 3,
            Some(what) => panic!("path: {} is wrong, because {}", path, what),
            None => panic!("path: {} is wrong, because None", path),
        };
        (label, name)
    }
    fn list_dir<P: AsRef<Path>>(&self, root: P) -> (Vec<PathBuf>, Vec<usize>, Vec<String>) {
        let root = Path::new(root.as_ref());
        assert!(root.exists());

        let mut images = vec![];
        let mut labels = vec![];
        let mut names = vec![];

        let pictures = walkdir::WalkDir::new(root)
            .into_iter()
            .filter(|p| p.is_ok())
            .filter(|p| p.as_ref().unwrap().path().is_file())
            .collect::<Vec<_>>();
        pictures
            .iter()
            .map(|pic| {
                let tmp_ps = pic.as_ref().unwrap().path();
                let ps = tmp_ps.to_str().unwrap();
                let (l, n) = self.get_label_name(ps);
                (tmp_ps.to_path_buf(), l, n)
            })
            .for_each(|(i, l, n)| {
                images.push(i);
                labels.push(l);
                names.push(n)
            });
        (images, labels, names)
    }
}

pub struct RGBDataloader {
    loader: Vec<Vec<(PathBuf, usize)>>,
    current_idx: usize,
    max_idx: usize,
    device: Device,
}
impl RGBDataloader {
    pub fn new(loader: Vec<Vec<(PathBuf, usize)>>, device: Device) -> Self {
        let max_idx = loader.len() - 1;
        let current_idx: usize = 0;
        Self {
            loader,
            current_idx,
            max_idx,
            device,
        }
    }
}

impl Iterator for RGBDataloader {
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        let imglbl = self.loader[self.current_idx].clone();
        self.current_idx += 1;
        if self.current_idx > self.max_idx {
            return None;
        }

        let (images, labels) = imglbl
            .par_iter()
            .map(|(img, lbl)| {
                let img = image::io::Reader::open(img)
                    .unwrap()
                    .decode()
                    .map_err(candle_core::Error::wrap)
                    .unwrap()
                    .resize_to_fill(
                        RGB_IMAGE_SIZE as u32,
                        RGB_IMAGE_SIZE as u32,
                        image::imageops::FilterType::Triangle,
                    );
                let img = img.to_rgb8();
                let data = img.into_raw();
                (data, *lbl as u8)
            })
            .collect::<(Vec<Vec<u8>>, Vec<u8>)>();
        let samples = images.len();
        let images: Vec<u8> = images.into_iter().flatten().collect::<Vec<_>>();
        let images_tensor = Tensor::from_vec(
            images,
            (samples, RGB_IMAGE_SIZE, RGB_IMAGE_SIZE, RGB_CHANNELS),
            &self.device,
        )
        .unwrap()
        .permute((0, 3, 1, 2))
        .unwrap();
        // let mean = Tensor::new(&[0.485f32, 0.456, 0.406], &Device::Cpu)?.reshape((3, 1, 1))?;
        // let std = Tensor::new(&[0.229f32, 0.224, 0.225], &Device::Cpu)?.reshape((3, 1, 1))?;
        // (data.to_dtype(candle::DType::F32)? / 255.)?
        //     .broadcast_sub(&mean)?
        //     .broadcast_div(&std)
        let images_tensor = (images_tensor.to_dtype(DType::F32).unwrap() / 255.).unwrap();
        let labels_tensor = Tensor::from_vec(labels.to_vec(), (samples,), &self.device).unwrap();
        let labels_tensor = labels_tensor.to_dtype(DType::U32).unwrap();
        Some((images_tensor, labels_tensor))
    }
}
