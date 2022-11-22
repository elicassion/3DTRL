NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=56277 imagenet_train.py "$@"
#/homes/jishang/datasets/imagenet1k

