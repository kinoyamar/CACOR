# To download quickdraw dataset

```bash
cat _quickdraw_download_list.txt | gsutil -m cp -I ./quickdraw/
```

The above only downloads data used in this experiment.

For the full dataset and detailed descriptions,
please refer to:

- [The Quick, Draw! dataset github repo](https://github.com/googlecreativelab/quickdraw-dataset)
- [Sketch-RNN github repo](https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn)
- [Google cloud storage of the dataset](https://console.cloud.google.com/storage/quickdraw_dataset/sketchrnn)
