date=$(date '+%Y-%m-%d-%H:%M:%S')

file=test
folder=additive
# folder=base


mkdir -p $folder/log



pytest $folder/${file}.py  2>&1 | tee -a $folder/log/${date}-${folder}.log
