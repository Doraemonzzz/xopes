date=$(date '+%Y-%m-%d-%H:%M:%S')
file=test

folder=normalize

mkdir -p $folder/log

export TORCHDYNAMO_VERBOSE=1
export TORCH_LOGS="+dynamo"

python $folder/${file}.py  2>&1 | tee -a $folder/log/${date}.log
