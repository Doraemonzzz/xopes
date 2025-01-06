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
folder=normalize/layernorm

##### householder 250101
folder=householder

##### logsumexp 250103
folder=logsumexp

##### linear_cross_entropy 250104
folder=linear_cross_entropy/vocab_size
# folder=linear_cross_entropy/hidden_dim
folder=linear_cross_entropy/batch_size

##### cross_entropy 250105
folder=cross_entropy

mkdir -p $folder/log

CUDA_VISIBLE_DEVICES=1 python $folder/${file}.py  2>&1 | tee -a $folder/log/${date}.log
