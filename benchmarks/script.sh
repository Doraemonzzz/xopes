date=$(date '+%Y-%m-%d-%H:%M:%S')

file=benchmark
folder=logcumsumexp
folder=grpe
folder=lrpe/cosine
# folder=md_lrpe_cosine
# folder=flao/non_causal
# folder=flao/fal_non_causal
# folder=act

mkdir -p $folder/log

python $folder/${file}.py  2>&1 | tee -a $folder/log/${date}.log
