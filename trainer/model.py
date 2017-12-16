from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import sys

import tensorflow as tf
import argparse

from tensorflow.contrib.learn.python.learn.estimators.dnn_linear_combined import DNNLinearCombinedEstimator
from tensorflow.contrib.learn.python.learn.estimators.head import Head, multi_head, regression_head, multi_label_head
from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig
from tensorflow.python.framework import dtypes

WIDE_COLUMNS = [
    tf.feature_column.numeric_column("wvec", shape=[2, 3], dtype=dtypes.float32),
]

DEEP_COLUMNS = [
    tf.feature_column.numeric_column("dvec", shape=[2, 3], dtype=dtypes.float32),
]

INPUT_COLUMNS = WIDE_COLUMNS + DEEP_COLUMNS


def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.iteritems()
    }
    return tf.contrib.learn.InputFnOps(features, None, inputs)


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn
}


def build_estimator(config, hidden_units=None):
    # Wide columns and deep columns.
    wide_columns = WIDE_COLUMNS
    deep_columns = DEEP_COLUMNS

    heads = multi_head([
        regression_head(head_name="g_wvec", label_name="g_wvec", label_dimension=2),
        regression_head(head_name="g_dvec", label_name="g_dvec", label_dimension=2),
    ])

    return DNNLinearCombinedEstimator(
        heads,
        config=config,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units or [4 * 2],
        fix_global_step_increment_bug=True
    )


def generate_input_fn(shuffle=True,
                      batch_size=2):
    wvec = [
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
    ]
    dvec = [
        [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
        [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], ],
    ]
    g_wvec = [
        [6.0, 9.0],
        [12.0, 15.0],
    ]
    g_dvec = [
        [5.3851648071, 7.0710678119, ],
        [8.7749643874, 13.9283882772, ]
    ]

    features = {
        'wvec': tf.stack(wvec),
        'dvec': tf.stack(dvec),
    }

    goal = {
        'g_wvec': tf.stack(g_wvec),
        'g_dvec': tf.stack(g_dvec),
    }

    # This operation builds up a buffer of parsed tensors, so that parsing
    # input data doesn't block training
    # If requested it will also shuffle
    if shuffle:
        features = tf.train.shuffle_batch(
            features,
            batch_size,
            min_after_dequeue=2 * batch_size + 1,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True
        )
    else:
        features = tf.train.batch(
            features,
            batch_size,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True
        )

    return features, goal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--job_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()

    config = RunConfig(model_dir=args.job_dir)

    sys.stdout.write("build_estimator...");
    sys.stdout.flush()
    build_estimator(config)
    sys.stdout.write("done\n");
    sys.stdout.flush()

    sys.stdout.write("generate_input_fn...");
    sys.stdout.flush()
    generate_input_fn()
    sys.stdout.write("done\n");
    sys.stdout.flush()
