pip index versions spacy
python install spacy==3.7.0
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm


torchrun --nproc_per_node=4 -m src.train 2>&1 | tee output.log





