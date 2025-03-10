date=$(date '+%Y-%m-%d-%H:%M:%S')

file=benchmark
folder=logcumsumexp
folder=grpe
folder=lrpe/cosine/_1d
# folder=lrpe/cosine/_md
# folder=md_lrpe_cosine
# folder=flao/non_causal
# folder=flao/fal_non_causal
# folder=multinomial

##### normalize 241231
# folder=normalize/srms_norm
# folder=normalize/rms_norm
# folder=normalize/group_norm
# folder=normalize/layer_norm
folder=normalize/normalize_with_gate

# ##### householder 250101
# folder=householder

# ##### logsumexp 250103
# folder=logsumexp

# ##### linear_cross_entropy 250104
# folder=linear_cross_entropy/vocab_size
# folder=linear_cross_entropy/hidden_dim
# folder=linear_cross_entropy/batch_size

# ##### cross_entropy 250105
# folder=cross_entropy

# ##### element_wise_binary_op 250106
# folder=element_wise_binary_op

# ##### out_product_linear_recurrence 250107
# folder=oplr/no_decay
# folder=oplr/ddd

# ##### lrpe 250115
# folder=lrpe/cosine/_1d
# folder=lrpe/rotate/_1d

# ##### act 250116
# folder=act

# ##### cumsum 250130
# folder=cumsum

# ##### gate_linear 250201
# folder=gate_linear/hidden_dim
# folder=gate_linear/batch_size

# ##### lavd 250204
# folder=lightning_attention/vector_decay

# ##### lasd 250220
# folder=lightning_attention/scalar_decay

##### lape 250309
folder=lightning_attention/positional_encoding

mkdir -p $folder/log

export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=1

python $folder/${file}.py  2>&1 | tee -a $folder/log/${date}.log
