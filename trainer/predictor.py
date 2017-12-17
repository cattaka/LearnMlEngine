import argparse
import multiprocessing
from datetime import datetime, date

from tensorflow.contrib.learn.python.learn.estimators import run_config
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils.input_fn_utils import InputFnOps
from tensorflow.contrib.predictor.predictor_factories import from_contrib_estimator
from tensorflow.core.protobuf.config_pb2 import ConfigProto

import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--job_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    # Argument to turn on all logging
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )

    args = parser.parse_args()

    run_config = run_config.RunConfig(model_dir=args.job_dir)
    estimator = model.build_estimator(run_config)


    def prediction_input_fn():
        wvec = [[1, 2, 3], [4, 5, 6]]
        dvec = [[2, 3, 4], [5, 6, 7]]
        features = {
            'wvec': tf.stack(wvec),
            'dvec': tf.stack(dvec),
        }
        inputs = {}
        for feat in model.INPUT_COLUMNS:
            inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

        return tf.contrib.learn.InputFnOps(features, None, inputs)


    predictor = from_contrib_estimator(
        estimator=estimator,
        prediction_input_fn=prediction_input_fn
    )
