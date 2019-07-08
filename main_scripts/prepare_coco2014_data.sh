#!/bin/bash

coco_dir=/nfs/data/cv/pub/mscoco/coco_2014

python extra_scripts/create_coco_data_files.py \
    --json_annotations_dir ${coco_dir}/annotations \
    --output_dir ${coco_dir} \
    --train_imgs_path ${coco_dir}/coco_train2014 \
    --val_imgs_path ${coco_dir}/coco_val2014
