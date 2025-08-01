# 1.1

## Added

- v1.1.1 Transformer: implemented by PyTorch, train in Multi30k dataset with single GPU successfully;
- v1.1.1 Transformer: compelete 'ddp' branch for DDP training;

## Changed

- v1.1.2 Transformer: save checkpoints only when bleu increases and loss decreases;
- v1.1.2 Transformer: adjust training hyper-parameters:
- v1.1.2 Transformer: updated 'requirements.txt', which removed unnecessary packages;
- v1.1.2 ~~Transformer: use mixed precision for training; [x] lead to higher Loss & PPL~~;

## Fixed

- v1.1.2 Transformer: replace 1e-9 by 'torch.finfo(attention_scores.dtype).min' to build mask;
- v1.1.2 Transformer: use 'skip_special_tokens=True' in tokenizer.decode for better code;
- v1.1.2 Transformer: fix echo logic in train.py, align info and only show some info in rank 0 process;




