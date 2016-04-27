#!/usr/bin/env sh
DATE=$(date +%Y_%m_%d_%H:%M:%S_)
# or you can use `dfdfdf` (from kk)

#echo $DATE) 

# make a sensible name
log_name='init'

if [ ! -d "log" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir log
fi

GLOG_logtostderr=1 \
th main.lua \
	-data /home/hongyang/dataset/imagenet_cls/cls \
	-nClasses 1000 \
	-nGPU 2 -nThreads 4 \
	-batchSize 32 -depth 18 \
	-shareGradInput 'true' \
	-tenCrop 'true' \
2>&1 | tee log/$DATE$log_name.log