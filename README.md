## SecondSight

### Getting started

#### bootstrap

```sh
$ ./script/bootstrap
```

#### activate python

```sh
$ . env/bin/activate
```

### Development

#### exec

```
$ python --job_dir workspace/output --train-steps 10000 --eval-steps 100
$ python trainer/predictor.py --job_dir workspace/output
```


### tensorboard
Known bug of virtualenv.
- https://github.com/dmlc/tensorboard/issues/36#issuecomment-322080363
- https://github.com/pypa/virtualenv/issues/355#issuecomment-318885792

Put "from distutils.sysconfig import get_python_lib" and replace "site.getsitepackages()" with "[get_python_lib()]".

```sh
$ tensorboard --logdir=workspace/output
```


### Train with MLEngine

refs https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction

#### Local

```sh
gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --train-steps 100000 \
    --eval-steps 100 \
    --job-dir workspace/output 
```

#### Remote

```sh
REGION=us-central1
BUCKET_NAME="your_bucket_name"
JOB_NAME=learn_ml_engine_1
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.4 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --eval-steps 100 \
    --train-steps 1000000 \
    --verbosity DEBUG
```

### Predict

refs https://cloud.google.com/ml-engine/docs/how-tos/deploying-models

#### local

```sh
gcloud ml-engine local predict \
    --model-dir workspace/output/export/Servo/XXXXXXX/ \
    --json-instances predict_test.json
```

#### remote

```sh
$ gcloud ml-engine versions create "the_version" \
     --model "learn_ml_engine" \
     --origin $OUTPUT_PATH/export/Servo/XXXXXXXX
$ gcloud ml-engine predict \
     --model $OUTPUT_PATH \
     --json-instances predict_test.json
```
