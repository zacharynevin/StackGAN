import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.tpu import TPUEstimator as Estimator
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from src.config import config
import src.estimator as estimator

def main(_):
    tpu_grpc_url = None

    if config.use_tpu:
        tpu_grpc_url = TPUClusterResolver(tpu=config.tpu_name).get_master()

    run_config = tpu.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=config.log_dir,
        session_config=tf.ConfigProto(allow_soft_placement=True),
        tpu_config=tpu.TPUConfig(config.tpu_iterations, config.tpu_shards)
    )

    batch_size = config.batch_size * config.tpu_shards if config.use_tpu else config.batch_size
    est = Estimator(
        model_fn=estimator.model_fn,
        use_tpu=config.use_tpu,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        params={
            "use_tpu": config.use_tpu,
            "data_dir": config.data_dir,
            "buffer_size": config.buffer_size,
            "data_format": "NCHW" if config.use_tpu else "NHWC",
            "z_dim": config.z_dim,
            "D_lr": config.d_lr,
            "G_lr": config.g_lr,
            "data_seed": config.data_shuffle_seed,
            "data_map_parallelism": config.data_map_parallelism
        },
        config=run_config
    )

    if config.train:
        est.train(
            input_fn=estimator.train_input_fn,
            max_steps=config.train_steps
        )
    if config.eval:
        est.evaluate(
            input_fn=estimator.eval_input_fn,
            steps=config.eval_steps
        )
    elif config.predict:
        est.predict(
            input_fn=lambda params: estimator.predict_input_fn(params, config.predict_class),
            predict_keys=['G2']
        )

if __name__ == '__main__':
    tf.app.run()
