# bert tokenizer, with DDP
(
    time torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 -m src.train \
        --de_tokenizer="bert-base-german-dbmdz-cased" \
        --en_tokenizer="bert-base-uncased" \
        --checkpoint_dir="./outs/bert-with-ddp/checkpoints" \
        --train_trace_dir="./outs/bert-with-ddp/train_traces" \
        --save_sample_batch_num=1 \
        --sample_trace_dir="./outs/bert-with-ddp/sample_traces" \
        --ddp
) 2>&1 | tee ./logs/bert-with-ddp.log

# bert tokenizer, without DDP
(
    {
        time python -u -m src.train \
            --de_tokenizer="bert-base-german-dbmdz-cased" \
            --en_tokenizer="bert-base-uncased" \
            --checkpoint_dir="./outs/bert-no-ddp/checkpoints" \
            --train_trace_dir="./outs/bert-no-ddp/train_traces" \
            --save_sample_batch_num=1 \
            --sample_trace_dir="./outs/bert-no-ddp/sample_traces"
    } 2>&1

) | tee ./logs/bert-no-ddp.log

# spacy tokenizer, with DDP
(
    time torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 -m src.train \
        --de_tokenizer="de_core_news_sm" \
        --en_tokenizer="en_core_web_sm" \
        --checkpoint_dir="./outs/spacy-with-ddp/checkpoints" \
        --train_trace_dir="./outs/spacy-with-ddp/train_traces" \
        --save_sample_batch_num=1 \
        --sample_trace_dir="./outs/spacy-with-ddp/sample_traces" \
        --ddp
) 2>&1 | tee ./logs/spacy-with-ddp.log

# spacy tokenizer, without DDP
(
    {
        time python -u -m src.train \
            --de_tokenizer="de_core_news_sm" \
            --en_tokenizer="en_core_web_sm" \
            --checkpoint_dir="./outs/spacy-no-ddp/checkpoints" \
            --train_trace_dir="./outs/spacy-no-ddp/train_traces" \
            --save_sample_batch_num=1 \
            --sample_trace_dir="./outs/spacy-no-ddp/sample_traces"
    } 2>&1

) | tee ./logs/spacy-no-ddp.log

