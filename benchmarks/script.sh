date=$(date '+%Y-%m-%d-%H:%M:%S')

file=benchmark
folder=logcumsumexp
folder=grpe
folder=lrpe_cosine
# folder=md_lrpe_cosine
# folder=flao_non_causal
folder=fuse_act_lrpe_non_causal

mkdir -p $folder/log

python $folder/${file}.py  2>&1 | tee -a $folder/log/${date}-${folder}.log
