# rgb-deepmae-candle

Deep learning training for RGB image classification using [candle](https://github.com/huggingface/candle).

## Example
```
cargo run -- --name 1031 --epochs 20 --batch-size 2
```

### command line

```
Usage: rgb-deepmae-candle [OPTIONS] --name <NAME>

Options:
  -n, --name <NAME>                    Name of this train
  -d, --dataset <DATASET>              specity WhichDataset [default: original] [possible values: original, l-enhanced, b-enhanced, lb-enhanced]
  -m, --model <MODEL>                  specity WhichModel [default: deep-maec16] [possible values: deep-maec16, deep-maec23, deep-maec28, deep-maec33]
  -l, --learning-rate <LEARNING_RATE>  learning rate [default: 0.05]
  -b, --batch-size <BATCH_SIZE>        batch size [default: 16]
  -e, --epochs <EPOCHS>                epochs [default: 10]
      --start-epoch <START_EPOCH>      start epochs [default: 1]
  -s, --seed <SEED>                    seed [default: 42]
      --load                           load checkpoint
      --save                           save checkpoint
  -r, --recoder-home <RECODER_HOME>    recoder home path [default: tmp]
  -h, --help                           Print help
  -V, --version                        Print version
```

