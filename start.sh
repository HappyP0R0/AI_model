#!/bin/sh
#PBS -e /mnt/home/abstrac01/krisjanis_elksnitis/logs
#PBS -o /mnt/home/abstrac01/krisjanis_elksnitis/logs
#PBS -q batch
#PBS -l nodes=1:ppn=8:gpus=1,feature=v100
#PBS -l mem=40gb
#PBS -l walltime=3:00:00:00

# max mem= 5 *ppn

module load conda
eval "$(conda shell.bash hook)"
conda activate krisjanis_elksnitis

cd /mnt/home/abstrac01/krisjanis_elksnitis
python ./DriverDetectionModel.py -run_name D1ToD1NoPooling -epoch 100 -use_pooling false -training_file ./100Drivers/data-splits/Traditional-setting/Day/Cam1/D1_train.txt -test_file ./100Drivers/data-splits/Traditional-setting/Day/Cam1/D1_val.txt -img_path_train ./100Drivers/Day/Cam1/ -img_path_test ./100Drivers/Day/Cam1/
python ./DriverDetectionModel.py -run_name D1ToN1NoPooling -epoch 100 -use_pooling false -training_file ./100Drivers/data-splits/Cross-modality-setting/D1_to_N1/D1_train.txt -test_file ./100Drivers/data-splits/Cross-modality-setting/D1_to_N1/N1_test.txt -img_path_train ./100Drivers/Day/Cam1/ -img_path_test ./100Drivers/Night/Cam1/
python ./DriverDetectionModel.py -run_name D1ToD1Pooling -epoch 100 -use_pooling true -training_file ./100Drivers/data-splits/Traditional-setting/Day/Cam1/D1_train.txt -test_file ./100Drivers/data-splits/Traditional-setting/Day/Cam1/D1_val.txt -img_path_train ./100Drivers/Day/Cam1/ -img_path_test ./100Drivers/Day/Cam1/
python ./DriverDetectionModel.py -run_name D1ToN1Pooling -epoch 100 -use_pooling true -training_file ./100Drivers/data-splits/Cross-modality-setting/D1_to_N1/D1_train.txt -test_file ./100Drivers/data-splits/Cross-modality-setting/D1_to_N1/N1_test.txt -img_path_train ./100Drivers/Day/Cam1/ -img_path_test ./100Drivers/Night/Cam1/
