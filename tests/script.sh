date=$(date '+%Y-%m-%d-%H:%M:%S')

file=test
folder=additive
folder=base
folder=logcumsumexp
folder=grpe
folder=lrpe/cosine/_md
# folder=lrpe/cosine/_md_act
# folder=flao/non_causal
# folder=flao/fal_non_causal
#
# folder=multinomial
folder=page_flip
folder=page_flip/additive

##### normalize 241231 / 250110 fix bug
folder=normalize/normalize
folder=normalize/normalize_fuse_residual # only pass fp32
folder=normalize/srms_norm
folder=normalize/rms_norm
folder=normalize/layer_norm
folder=normalize/group_norm
folder=normalize/group_rms_norm
folder=normalize/group_srms_norm
folder=normalize/normalize_fuse_gate # 250301 add

# ##### householder 250101
# folder=householder

# ##### logsumexp 250103
# folder=logsumexp

# ##### linear_cross_entropy 250105
# folder=linear_cross_entropy

# ##### cross_entropy 250106
# folder=cross_entropy

# ##### element_wise_binary_op 250106
# folder=element_wise_binary_op
# folder=element_wise_binary_op/fn

# ##### oplr 250109
# folder=out_product_linear_recurrence/data_dependent_decay

# ##### lrpe 250114
# folder=lrpe/cosine/_1d
# folder=lrpe/cosine/_1d_mpa
# folder=lrpe/rotate/_1d
# folder=lrpe/rotate/_1d_mpa

# ##### act 250116
# folder=act

# ##### lce 250119
folder=linear_cross_entropy

##### cumsum 250129
# folder=cumsum/cumsum

# ##### gate_linear 250201
# folder=gate_linear

# ##### lacd 250219
# folder=lightning_attn/constant_decay

# ##### lacd_part 250224
# folder=lightning_attn/constant_decay/intra
# folder=lightning_attn/constant_decay/state
# folder=lightning_attn/constant_decay/inter

# ##### lasd 250303
folder=lightning_attn/scalar_decay
# folder=lightning_attn/scalar_decay/intra
# folder=lightning_attn/scalar_decay/state
# folder=lightning_attn/scalar_decay/inter

# ##### chunk_cumsum 250305
# folder=cumsum/chunk_cumsum

# ##### lape 250309
# folder=lightning_attn/positional_encoding

# ##### lcse 250312
# folder=logcumsumexp

# ##### dld 250317
# folder=lightning_attn/log_decay/dld
folder=lightning_attn/log_decay/dld_with_cumsum

# ##### chunk_cumsum_reduce 250318
# folder=cumsum/chunk_cumsum_reduce
# folder=cumsum/chunk_cumsum
# folder=cumsum/chunk_reverse_cumsum
# folder=cumsum/chunk_cumsum_decay
# folder=cumsum/cumsum

# ##### lavd 250122, 250403
folder=lightning_attn/vector_decay
# folder=lightning_attn/vector_decay/state
# folder=lightning_attn/vector_decay/inter
# folder=lightning_attn/vector_decay/intra
folder=lightning_attn/vector_decay/recurrence
# folder=lightning_attn/vector_decay/sub_intra

# ##### laer 250407
# folder=lightning_attn/element_recurrence
# folder=lightning_attn/element_recurrence/state

# ##### tpa_decode 250408
# folder=flash_attn/tpa_decode

# ##### inverse 250502
# folder=inverse

##### ilav 250525
folder=implicit_attn/inverse_attn

# ##### poly_attn 250610
# folder=poly_attn

##### ladd 250613
folder=lightning_attn/delta_decay/inv

##### krcl 250617
folder=kernel_regression/causal_linear
# folder=kernel_regression/causal_linear/inv

mkdir -p $folder/log


export XOPES_DEBUG=True
export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
# export TRITON_INTERPRET=1
export TRITON_F32_DEFAULT=ieee

pytest $folder/${file}.py  2>&1 | tee -a $folder/log/${date}.log
