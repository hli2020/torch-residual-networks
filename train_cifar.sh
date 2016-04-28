#!/usr/bin/env sh
DATE=$(date +%Y_%m_%d_%H:%M:%S_)
# or you can use `dfdfdf` (from kk)

#echo $DATE) 

# make a sensible name
log_name='s190_cifar'

if [ ! -d "log" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir log
fi

#-data /home/hongyang/dataset/imagenet_cls/cls \
GLOG_logtostderr=1 \
CUDDA_VISIBLE_DEVICES=9,10 \
th main.lua \
	-data /data2/dataset/cifar-10-batches-t7 \
	-nClasses 10 \
	-dataset cifar10 \
	-nGPU 2 \
	-batchSize 128 -depth 1202 \
	-shareGradInput 'true' \
	-tenCrop 'true' \
2>&1 | tee log/$DATE$log_name.log
