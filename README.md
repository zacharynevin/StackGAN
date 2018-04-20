# StackGAN

Tensorflow implementation of the StackGAN++ outlined in this paper: [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1710.10916.pdf).

This implementation uses the Estimator API, allowing you to train StackGAN++ models on novel datasets with minimal effort.

### Features

- Easy to use. To retrain this model, simply run `python3 main.py --train --data_dir=/path/to/your/dataset --log_dir=/path/to/logs`. See the full list of [configuration options](https://google.com).
- Fully TPU compatible. To run on gcloud TPUs, use the `--tpu_name` flag.
- Native Tensorboard integration.

### Getting Started

#### Create Dataset

You must format your dataset as a TFRecord file with the given features:
```
image   : bytes
height  : int64
width   : int64
channels: int64
label   : int64
```

An example is shown below:

```python
import tensorflow as tf
import cv2

writer = tf.python_io.TFRecordWriter("/path/to/my/dataset.tfrecords")

for f in os.listdir("./raw_images"):
  img = cv2.imread(os.path.join("./raw_images", f))
  height, width, channels = img.shape
  class_index = 0 # 0 to (num_classes-1)

  example = tf.train.Example(
    features = tf.train.Features(
      feature = {
        "image_64": _bytes_feature(resize(img, 64).tostring()),
        "image_128": _bytes_feature(resize(img, 128).tostring()),
        "image_256": _bytes_feature(resize(img, 256).tostring()),
        "label": _int64_feature(class_index)
      }
    )
  )

  writer.write(example.SerializeToString())
```

Once this `.tfrecords` file is created, you can immediately use it to train your model from your local machine. Alternatively, you can upload it to a gcloud storage bucket that you own and reference it from there, which is advantageous if you are using AWS or Gcloud VMs and don't want to worry about a time-consuming process of downloading the dataset first.

*NOTE*: Using a google storage location for your dataset and log files is a requirement when using TPUs.

#### Run Training

Training the model on a new dataset is easy. Locally, you can just run:

```
python3 main.py --train --data_dir=/path/to/dataset.tfrecords
```

If your dataset is stored on gcloud storage, you can simply replace with `data_dir` with a fully qualified google storage path:

```
python3 main.py --train --data_dir=gs://${BUCKET}/path/to/dataset.tfrecords
```

#### Monitor

To monitor training, point a `tensorboard` instance to your log dir:

```
tensorboard --logdir=/path/to/logs
```

### Using TPUs

This repos is written to be fully TPU compatible. Assuming you have already provisioned a TPU on gcloud, you can use the `--tpu_name` flag:

```
python3 main.py --train --data_dir=gs://${BUCKET}/path/to/dataset.tfrecords --tpu_name=${TPU_NAME}
```

### Configuration Options

```
- data_dir: String. The data directory. Must point to a tfrecords file. Can be a google storage path (e.g. gs://my-bucket/my/path/file.tfrecords).
```
