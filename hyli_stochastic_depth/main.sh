#!/usr/bin/env sh
DATE=$(date +%Y_%m_%d_%H:%M:%S_)
# or you can use `dfdfdf` (from kk)

#echo $DATE) 

# make a sensible name
log_name='stochastic'

if [ ! -d "log" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir log
fi

if [ ! -d "result" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir result
fi

#-data /home/hongyang/dataset/imagenet_cls/cls \
GLOG_logtostderr=1 \
th main.lua \
	-dataRoot /home/hongyang/dataset/cifar.torch/ \
	-resultFolder result \
	-device 0 \
	-N 18 \
	-deathRate 0.5 \
	-deathMode 'lin_decay' \
2>&1 | tee log/$DATE$log_name.log