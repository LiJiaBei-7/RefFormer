# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --dist 1 --task itr_refcoco --config configs/cclm-base-ft/Retrieval_refcocog.yaml --output_dir output/path/to/save/refcocog/CLIP_finetune --bs 512 --seed 42 --epoch 10 --checkpoint /mnt/workspace/CCR2/xlmr/data/cclm_3m_epoch_29.th

# dt_refcoco itr_refcoco gd_refcoco_poster gd_refcoco text_region_refcoco region_refcoco
# 0,1,2,3,4,5,6,7

# output/refcocj/refcoco_faster_base/checkpoint_best.pth
# bridger_6_2_4_aux*0.1
# sleep 3.5h
#  output/refcocog/bridger_6_2_5_0.1aux_enhance
# sleep 2h
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run.py \
    --dist l4 --task gd_refcoco --config configs/cclm-base-ft/Retrieval_refcocog.yaml \
    --output_dir output/refcocog/bridger_6_2_5_2_query3_0.1aux_521_gate_share --bs 32 --seed 100 --epoch 50 --checkpoint null