CUDA_VISIBLE_DEVICES="" python evaluator.py \
    ./cifar10_ref.npz \
    ./log_dir/cifar10_1/baseline10kiter_samples_50000x32x32x3.npz \
    --log_dir ./log_dir/cifar10_1