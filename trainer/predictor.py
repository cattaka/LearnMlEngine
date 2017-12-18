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
        feature_placeholders = {
            'wvec': tf.placeholder(tf.float32, [1, 2, 3]),
            'dvec': tf.placeholder(tf.float32, [1, 2, 3]),
        }
        features = {
            key: tf.expand_dims(tensor, -1)
            for key, tensor in feature_placeholders.items()
        }

        return tf.contrib.learn.InputFnOps(features, None, feature_placeholders)


    predictor = from_contrib_estimator(
        estimator=estimator,
        prediction_input_fn=prediction_input_fn,
        output_alternative_key="g_dvec"
    )

    sess = tf.Session()
    print(predictor.fetch_tensors)
#    print(predictor.fetch_tensors["scores"].eval())
    wvec = [[[1, 2, 3], [4, 5, 6]]]
    dvec = [[[2, 3, 4], [5, 6, 7]]]
    i = 0
    g_wvec = [
        [ model.dist(wvec[i][0], wvec[i][1]),  model.dist(dvec[i][0], dvec[i][1])],
    ]
    g_dvec = [
        [ model.dist_r(wvec[i][0], wvec[i][1], dvec[i][0], dvec[i][1]),  model.dist_r(wvec[i][0], dvec[i][1], wvec[i][0], dvec[i][1])],
    ]

    features = {
            'wvec': wvec,
            'dvec': dvec,
        }
    print(predictor(features))
    print([g_wvec, g_dvec])
