#!/usr/bin/env sh
DATE=$(date +%Y_%m_%d_%H:%M:%S_)
# or you can use `dfdfdf` (from kk)

#echo $DATE) 

# make a sensible name
log_name='s190'

if [ ! -d "log" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir log
fi

#-data /home/hongyang/dataset/imagenet_cls/cls \
GLOG_logtostderr=1 \
th main.lua \
	-data /data2/dataset/imagenet_cls/raw \
	-nClasses 500 \
	-nGPU 8 -nThreads 8 \
	-batchSize 256 -depth 18 \
	-shareGradInput 'true' \
	-tenCrop 'true' \
2>&1 | tee log/$DATE$log_name.log