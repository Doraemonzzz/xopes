date=$(date '+%Y-%m-%d-%H:%M:%S')

file=benchmark
folder=logcumsumexp
folder=grpe

mkdir -p $folder/log

python $folder/${file}.py  2>&1 | tee -a $folder/log/${date}-${folder}.log
