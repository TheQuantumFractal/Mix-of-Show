# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B4_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B3_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B2_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B1_Repeat500.yml
# accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B0_Repeat500.yml

accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B4_1e-6_1e-4_Repeat500.yml
accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B3_1e-6_1e-4_Repeat500.yml
accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B2_1e-6_1e-4_Repeat500.yml
accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B1_1e-6_1e-4_Repeat500.yml
accelerate launch  train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRSA_potter_Cmix_B0_1e-6_1e-4_Repeat500.yml


bash /home/sarafk/Mix-of-Show/evaluate_total.sh