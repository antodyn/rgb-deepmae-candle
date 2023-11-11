#[derive(Debug, Copy, Clone, PartialEq)]
struct MyNanKey(f32);

impl Eq for MyNanKey {}

impl PartialOrd for MyNanKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MyNanKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => self.0.partial_cmp(&other.0).unwrap(),
        }
    }
}
// https://stackoverflow.com/questions/40408293/how-do-i-sort-nan-so-that-it-is-greater-than-any-other-number-and-equal-to-an
pub fn float_argsort(data: &Vec<f32>) -> Vec<u32> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| MyNanKey(data[i]));
    indices.into_iter().map(|i| i as u32).collect::<Vec<_>>()
}
pub fn int_argsort<T: Ord>(data: &[T]) -> Vec<u32> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices.into_iter().map(|i| i as u32).collect::<Vec<_>>()
}
