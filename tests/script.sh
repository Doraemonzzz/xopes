date=$(date '+%Y-%m-%d-%H:%M:%S')

file=test
folder=additive
folder=base
folder=logcumsumexp
folder=grpe
folder=lrpe/cosine/_1d
folder=lrpe/cosine/_md
# folder=lrpe/cosine/_md_act
# folder=flao/non_causal
# folder=flao/fal_non_causal
# folder=act
# folder=multinomial
folder=page_flip
folder=page_flip/additive

##### normalize 241231
folder=normalize/normalize
# folder=normalize/normalize_fuse_residual
# folder=normalize/srmsnorm
# folder=normalize/rmsnorm
# folder=normalize/layernorm

##### householder 250101
# folder=householder

##### logsumexp 250103
# folder=logsumexp

##### linear_cross_entropy 250105
folder=linear_cross_entropy

##### cross_entropy 250106
folder=cross_entropy

##### element_wise_binary_op 250106
folder=element_wise_binary_op
# folder=element_wise_binary_op/fn

mkdir -p $folder/log


export XOPES_DEBUG=True
# export TRITON_INTERPRET=1

pytest $folder/${file}.py  2>&1 | tee -a $folder/log/${date}.log
