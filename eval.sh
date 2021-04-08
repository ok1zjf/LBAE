#!/bin/bash

# Must be run from the experiment directory
# Latest training weights will be used 

SBIN_ROOT="../"  # Path to the python scripts
CD=$(pwd)
dir_name=$(basename "$CD")
toks=(${dir_name//_/ })
dataset_name=${toks[0]}
echo "Dataset name:" ${dataset_name}

TEST_IMGS_DIR=../${dataset_name}_eval/
img_num=`ls $TEST_IMGS_DIR/*.jpg | wc -l`
echo "Test images directory: $TEST_IMGS_DIR"
echo "Number of test images: $img_num"

#VAE_data=../mnist_vae_res32_ConvResBlock32-vae-z16-m169/generated/samples_random/
#VAE_label=VAE

#export CUDA_VISIBLE_DEVICES=1 
python3 ${SBIN_ROOT}${dataset_name}.py l eval
python3 ${SBIN_ROOT}${dataset_name}.py l eval gen sample_method=cov gen_imgs=${img_num}
python3 ${SBIN_ROOT}${dataset_name}.py l eval gen sample_method=random gen_imgs=${img_num}
python3 ${SBIN_ROOT}${dataset_name}.py l eval gen sample_method=int gen_imgs=${img_num} batch_size_test=10 interpolate_steps=10

# Empty the precision-recall cache
rm -f /tmp/prd_cache/*
python3 ../../precision-recall-distributions/prd_from_image_folders.py --inception_path ../../precision-recall-distributions/model/classify_image_graph_def.pb --reference_dir ../${dataset_name}_eval/ --eval_dirs generated/samples_cov/ generated/samples_random/ ${VAE_data} --eval_labels cov rnd ${VAE_label} --plot_path PR.pdf

python3 ${SBIN_ROOT}eval.py s=../${dataset_name}_eval t=generated/samples_cov,generated/samples_int,generated/samples_random,reco

