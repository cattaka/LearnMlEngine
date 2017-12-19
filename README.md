## SecondSight

### bootstrap

```sh
$ ./script/bootstrap
```

### activate python

```sh
$ . env/bin/activate
```

### exec

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
