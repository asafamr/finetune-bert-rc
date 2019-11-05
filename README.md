# finetune-bert-rc
training BERT for relation classfication using "matching the blanks" recommanded method

overview of repo files:
* datautils.py data loading primitives 
* simplifying_wrapper.py simple wrapping methods to full finetuning code
* **ftbert.ipynb** a notebook implementing a dataloader for our spike jsons and then training roberta with it
* rc_transformer.py a pytorch model - matching the blanks-like RC with a transformer
* transformers_rc_finetune.py - all the training boilerplate together with some sane training settings
