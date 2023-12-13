# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B4_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B3_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B2_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B1_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B0_Repeat500.yml

# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B4_1e-6_1e-4_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B3_1e-6_1e-4_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B2_1e-6_1e-4_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B1_1e-6_1e-4_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-6_1e-4_Repeat500.yml

# Takeaway: these ones didn't get params truly to 0, and didn't improve quality. maybe need higher threshold?
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B4_1e-6_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B3_1e-6_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B2_1e-6_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B1_1e-6_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-6_0_Repeat500.yml

# marginally better, but still need higher threshold
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B4_1e-5_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B3_1e-5_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B2_1e-5_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B1_1e-5_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-5_0_Repeat500.yml

# 1e-4 is too high; basically kills everything
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B4_1e-4_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B3_1e-4_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B2_1e-4_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B1_1e-4_0_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-4_0_Repeat500.yml


# for shrink_thresh in 2e-5 4e-5 6e-5 8e-5;
# do
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B4_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B3_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B2_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B1_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
# done

# for l1 in 3e-5 3e-4;
# do
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B4_1e-6_1e-4_Repeat500.yml -l1 $l1
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B3_1e-6_1e-4_Repeat500.yml -l1 $l1
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B2_1e-6_1e-4_Repeat500.yml -l1 $l1
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B1_1e-6_1e-4_Repeat500.yml -l1 $l1
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-6_1e-4_Repeat500.yml -l1 $l1
# done

# for seed in 1 2 3 4;
# do
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B0_Repeat500.yml -seed $seed
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B1_Repeat500.yml -seed $seed
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B2_Repeat500.yml -seed $seed
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B3_Repeat500.yml -seed $seed
#     accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B4_Repeat500.yml -seed $seed
# done

# Run just the rank 0 hermione
# for character in hermione;
# do
#     accelerate launch train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-6_1e-4_Repeat500.yml -character $character
#     accelerate launch train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B1_Repeat500.yml -character $character
# done

# Focus on rank 0 LoRSA and compare sparsity params
for shrink_thresh in 4e-7 6e-7 8e-7 1e-6 2e-6 3e-6 4e-6 5e-6;  # default is 1e-6, higher means more sparse. tried increments of 10x to no avail.
do
    for l1 in 0 4e-5 6e-5 8e-5 1e-4 2e-4 3e-4 4e-4 5e-4;  # default is 1e-4, higher means more sparse. tried increments of 10x to no avail.
    do
        accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-6_1e-4_Repeat500.yml -shrinkage $shrink_thresh -l1 $l1 -soft
    done
done

bash /home/sarafk/Mix-of-Show/evaluate_total.sh