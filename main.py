import tensorflow as tf
import src.train
import src.config as config

def main(_):
    src.train.main(_)

if __name__ == '__main__':
    tf.app.run()
