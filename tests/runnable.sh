# bert tokenizer, with DDP
(
    time torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 -m src.train \
        --de_tokenizer="bert-base-german-dbmdz-cased" \
        --en_tokenizer="bert-base-uncased" \
        --checkpoint_dir="./outs/bert-with-ddp/checkpoints" \
        --train_trace_dir="./outs/bert-with-ddp/train_traces" \
        --save_sample_batch_num=3 \
        --sample_trace_dir="./outs/bert-with-ddp/sample_traces" \
        --ddp
) 2>&1 | tee ./logs/bert-with-ddp.log

