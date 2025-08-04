# 1.1

## Added

**v1.1.3**:
- v1.1.3 updated 'CHANGELOG.md' and 'README.md';
- v1.1.3 added spacy tokenizer;

**v1.1.1**:
- v1.1.1 implemented by PyTorch, train in Multi30k dataset with single GPU successfully;
- v1.1.1 compelete 'ddp' branch for DDP training;

## Changed

**v1.1.3**:
- v1.1.3 save checkpoints only when bleu increases;
- v1.1.3 changed some variables name in transformer;
- v1.1.3 adjust training hyper-parameters;
- v1.1.3 updated 'requirements.txt', which added more necessary packages;

**v1.1.2**:
- v1.1.2 updated 'requirements.txt', which removed unnecessary packages;
- v1.1.2 adjust training hyper-parameters;
- v1.1.2 ~~use mixed precision for training; [x] lead to higher Loss & PPL~~;

## Fixed

**v1.1.3**:
- v1.1.3 use auto-regressive for evaluating;
- v1.1.3 fix bleu;

**v1.1.2**:
- v1.1.2 replace 1e-9 by 'torch.finfo(attention_scores.dtype).min' to build mask;
- v1.1.2 use 'skip_special_tokens=True' in tokenizer.decode for better code;
- v1.1.2 fix echo logic in train.py, align info and only show some info in rank 0 process;





