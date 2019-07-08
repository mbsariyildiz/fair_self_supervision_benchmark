#!/bin/bash

data_dir=/slow_data/bulentsariyildiz/datasets/voc2007
params_file=https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/caffenet_bvlc_in1k_supervised.npy
features_dir=/slow_data/bulentsariyildiz/experiments/ssl-benchmark/alexnet-in1k/voc2007/features
svm_output_dir=/slow_data/bulentsariyildiz/experiments/ssl-benchmark/alexnet-in1k/voc2007/svm

mkdir -p ${features_dir}
mkdir -p ${svm_output_dir}
mkdir -p ${svm_output_dir}/conv1
mkdir -p ${svm_output_dir}/conv2
mkdir -p ${svm_output_dir}/conv3
mkdir -p ${svm_output_dir}/conv4
mkdir -p ${svm_output_dir}/conv5

# echo "Extracting features of the training set"
# python tools/extract_features.py \
#     --config_file configs/benchmark_tasks/image_classification/voc07/caffenet_bvlc_supervised_extract_features.yaml \
#     --data_type train \
#     --output_file_prefix trainval \
#     --output_dir ${features_dir} \
#     TEST.PARAMS_FILE ${params_file} \
#     TRAIN.DATA_FILE ${data_dir}/train_images.npy \
#     TRAIN.LABELS_FILE ${data_dir}/train_labels.npy

# echo "Extracting features of the test set"
# python tools/extract_features.py \
#     --config_file configs/benchmark_tasks/image_classification/voc07/caffenet_bvlc_supervised_extract_features.yaml \
#     --data_type test \
#     --output_file_prefix test \
#     --output_dir ${features_dir} \
#     TEST.PARAMS_FILE ${params_file} \
#     TEST.DATA_FILE ${data_dir}/test_images.npy \
#     TEST.LABELS_FILE ${data_dir}/test_labels.npy


layer_idx=5
feature_subset_name="conv5_s1k8"

# echo "Training an SVM on the features of the conv_${layer_idx} layer"
# python tools/svm/train_svm_kfold.py \
#     --data_file ${features_dir}/trainval_${feature_subset_name}_resize_features.npy \
#     --targets_data_file ${features_dir}/trainval_${feature_subset_name}_resize_targets.npy \
#     --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
#     --output_path ${svm_output_dir}/conv${layer_idx}


echo "Testing the SVM"
python tools/svm/test_svm.py \
  --data_file ${features_dir}/test_${feature_subset_name}_resize_features.npy \
  --targets_data_file ${features_dir}/test_${feature_subset_name}_resize_targets.npy \
  --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
    --output_path ${svm_output_dir}/conv${layer_idx}
