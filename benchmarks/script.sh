date=$(date '+%Y-%m-%d-%H:%M:%S')

file=benchmark
folder=logcumsumexp
folder=grpe
folder=lrpe/cosine/_1d
# folder=lrpe/cosine/_md
# folder=md_lrpe_cosine
# folder=flao/non_causal
# folder=flao/fal_non_causal
# folder=act
# folder=multinomial

##### normalize 241231
# folder=normalize/srmsnorm
# folder=normalize/rmsnorm
# folder=normalize/groupnorm
# folder=normalize/layernorm

##### householder 250101
folder=householder

mkdir -p $folder/log

python $folder/${file}.py  2>&1 | tee -a $folder/log/${date}.log
