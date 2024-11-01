#!/bin/sh -V
#PBS -e /mnt/home/abstrac01/krisjanis_elksnitis/logs
#PBS -o /mnt/home/abstrac01/krisjanis_elksnitis/logs
#PBS -q batch
#PBS -l nodes=1:ppn=12:gpus=1,feature=k40
#PBS -l mem=60gb
#PBS -l walltime=01:00:00

# max mem= 5 *ppn

cd /mnt/home/abstrac01/krisjanis_elksnitis
python ./DriverDetectionModel.py -run_name D1ToD1 -epoch 10 -training_file ./10Drivers/data-splits/Cross-modality-setting/D1_to_N1/D1_train.txt -test_file ./10Drivers/data-splits/Cross-modality-setting/D1_to_N1/D1_val.txt -img_path_train ./10Drivers/Day/Cam1 -img_path_test ./100Drivers/Day/Cam1
python ./DriverDetectionModel.py -run_name D1ToN1 -epoch 10 -training_file ./10Drivers/data-splits/Cross-modality-setting/D1_to_N1/D1_train.txt -test_file ./10Drivers/data-splits/Cross-modality-setting/D1_to_N1/N1_test.txt -img_path_train ./10Drivers/Day/Cam1 -img_path_test ./100Drivers/Night/Cam1