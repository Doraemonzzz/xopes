date=$(date '+%Y-%m-%d-%H:%M:%S')

file=test
folder=additive
# folder=base
folder=logcumsumexp

mkdir -p $folder/log


export XOPES_DEBUG=True

pytest $folder/${file}.py  2>&1 | tee -a $folder/log/${date}-${folder}.log
