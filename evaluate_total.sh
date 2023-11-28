printf 'ArcFace Score\n' >> output.txt

device=1
for filename in /home/kamyar/Mix-of-Show/experiments/*/visualization/PromptDataset/Iters-latest_Alpha-1.0;
do
    screen -dm bash -c "export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda; conda activate mix; CUDA_VISIBLE_DEVICES=$(($device % 8)) python /home/kamyar/evaluation.py /home/kamyar/Mix-of-Show/datasets/data/characters/real/Harry_Potter/image $filename >> output.txt"
    echo device
    if ((device % 8 == 0)); then
        sleep 600
    fi
    device=$((device+1))
done

printf 'Parameter Count\n' >> output.txt

device=0
for filename in /home/kamyar/Mix-of-Show/experiments/*/models/*.pth;
do
    screen -dm bash -c "conda activate mix; CUDA_VISIBLE_DEVICES=$(($device % 8)) python /home/kamyar/parameter_counter.py $filename >> output.txt"
    device=$((device+1))
done