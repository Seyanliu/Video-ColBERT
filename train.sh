export CUDA_VISIBLE_DEVICES=5

setsid torchrun \
    --nproc_per_node=1 \
    --master_port=29505 \
    main.py \
    --do_train 4 \
    --workers 8 \
    --n_display 50 \
    --epochs 5 \
    --lr 1e-4 \
    --coef_lr 1e-3 \
    --batch_size 256 \
    --batch_size_val 256 \
    --anno_path /data/liusiyuan/dataset/MSRVTT/anns \
    --video_path /data/liusiyuan/dataset/MSRVTT/videos \
    --datatype msrvtt \
    --max_words 32 \
    --max_frames 12 \
    --video_framerate 1 \
    --use_query_expansion \
    --base_encoder ViT-B/32 \
    --agg_module seqTransf \
    --interaction mms \
    --output_dir ckpts/msrvtt_videocolbert \
    > train_videocolbert.log 2>&1 &