PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


python train.py --config='configs/comparison_acdc_224_136/gcf_unet_r50.yaml' --device='cuda:1' \
                --work_dir='model'
