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


for shrink_thresh in 2e-5 4e-5 6e-5 8e-5;
do
    accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B4_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
    accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B3_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
    accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B2_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
    accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B1_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
    accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-4_0_Repeat500.yml -shrinkage $shrink_thresh
done

bash /home/sarafk/Mix-of-Show/evaluate_total.sh