pip index versions spacy
python install spacy==3.7.0
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm


torchrun --nproc_per_node=4 -m src.train 2>&1 | tee output.log
time torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 -m src.scripts.train 2>&1 | tee output.log

(time torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 -m src.scripts.train --de_tokenizer="de_core_news_sm" --en_tokenizer="en_core_web_sm" --checkpoint_dir="./checkpoints/spacy-with-ddp" --trace_dir="./outs/spacy-with-ddp" --ddp=True) 2>&1 | tee ./logs/spacy-with-ddp.log


