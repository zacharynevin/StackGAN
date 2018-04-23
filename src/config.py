import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_bool("train", False, "Run training [False].")
flags.DEFINE_bool("predict", False, "Run prediction [False].")
flags.DEFINE_bool("eval", False, "Run evaluation [False].")

flags.DEFINE_integer("predict_class", None, "The class to generate. If None, generate images from random classes [None]")

flags.DEFINE_bool("use_tpu", False, "Set to True to use TPUs [False].")
flags.DEFINE_string("tpu_name", None, "The name of the TPU to use [None].")
flags.DEFINE_integer("tpu_shards", 8, "Number of TPU shards [8].")
flags.DEFINE_integer("tpu_iterations", 50, "Number of iterations per TPU training loop [50].")

flags.DEFINE_string("data_dir", "./data/dataset.tfrecords", "The data directory. Must point to a tfrecords file. Can be a google storage path (e.g. gs://my-bucket/my/path/file.tfrecords).")
flags.DEFINE_string("log_dir", "./logs", "Directory to store logs. Can be a google storage path (e.g. gs://my-bucket/my/path).")

flags.DEFINE_integer("buffer_size", 8*1024*1024, "The dataset buffer size. [8388608]")
flags.DEFINE_integer("batch_size", 128, "The batch size. If using TPUs, this is the batch size per shard. [64]")
flags.DEFINE_integer("z_dim", 100, "The z input dimension [100].")
flags.DEFINE_integer("data_shuffle_seed", 12345, "The seed to use when shuffling the database [12345].")
flags.DEFINE_integer("data_map_parallelism", 10, "The number of parallel calls to use in dataset.map [10].")

flags.DEFINE_float("g_lr", 0.0002, "The generator learning rate [2e-4].")
flags.DEFINE_float("d_lr", 0.0002, "The discriminator learning rate [2e-4].")

flags.DEFINE_integer("train_steps", 1000, "The number of training steps [1000].")
flags.DEFINE_integer("eval_steps", 1000, "The number of eval steps [1000].")

config = flags.FLAGS

if config.train and config.tpu_name:
    config.use_tpu = True
    config.data_format = 'NCHW'
