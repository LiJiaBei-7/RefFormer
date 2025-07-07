
CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --dist f1 --task gd_refcoco --config configs/grounding/grounding_refcoco_mix.yaml \
    --output_dir /mnt_rela/wangyabing.wyb/outputs/vg_bridge/grounding/refcoco_mix/exp1 --bs 32 --seed 100 --epoch 50 --checkpoint null



# mix
# CUDA_VISIBLE_DEVICES=0,1 python3 run.py \
#     --dist f2 --task gd_refcoco --config configs/grounding/grounding_refcoco_mix.yaml \
#     --output_dir /mnt_rela/wangyabing.wyb/outputs/vg_bridge/grounding/refcoco_mix/exp1 --bs 32 --seed 100 --epoch 50 --checkpoint null


# referIt
# CUDA_VISIBLE_DEVICES=0,1 python3 run.py \
#     --dist f2 --task gd_referIt --config configs/grounding/grounding_referIt.yaml \
#     --output_dir /mnt_rela/wangyabing.wyb/outputs/vg_bridge/grounding/referIt/exp1  --bs 32 --seed 100 --epoch 50 --checkpoint null


# flickr30k
# CUDA_VISIBLE_DEVICES=0,1 python3 run.py \
#     --dist f2 --task gd_flickr30k --config configs/grounding/grounding_flickr30k.yaml \
#     --output_dir /mnt_rela/wangyabing.wyb/outputs/vg_bridge/grounding/flickr30k/exp1  --bs 32 --seed 100 --epoch 50 --checkpoint null


# refcocog
# CUDA_VISIBLE_DEVICES=0,1 python3 run.py \
#     --dist f2 --task gd_refcoco --config configs/grounding/grounding_refcocog.yaml \
#     --output_dir /mnt_rela/wangyabing.wyb/outputs/vg_bridge/grounding/refcocog/exp1 --bs 32 --seed 100 --epoch 50 --checkpoint null


# refcoco
# CUDA_VISIBLE_DEVICES=0,1 python3 run.py \
#     --dist f2 --task gd_refcoco --config configs/grounding/grounding_refcoco.yaml \
#     --output_dir /mnt_rela/wangyabing.wyb/outputs/vg_bridge/grounding/refcoco/exp1 --bs 32 --seed 100 --epoch 50 --checkpoint null


# refcoco+
# CUDA_VISIBLE_DEVICES=0,1 python3 run.py \
#     --dist f2 --task gd_refcoco --config configs/grounding/grounding_refcoco+.yaml \
#     --output_dir /mnt_rela/wangyabing.wyb/outputs/vg_bridge/grounding/refcoco+/exp1 --bs 32 --seed 100 --epoch 50 --checkpoint null
