
# 最后再统一把PAD  BERT_PAD 合并
# PAD = 0
# UNK = 1
# BOS = 2
# EOS = 3

# PAD_WORD = '<blank>'
# UNK_WORD = '<unk>'
# BOS_WORD = '<s>'
# EOS_WORD = '</s>'

#by me
PAD = 0
UNK = 100
BOS = 101
EOS = 102

PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

checkpointing = 0
static = False
residual_type = 'regular'
max_position_length = 8192

#by me
BERT_PAD_WORD = '[PAD]'
BERT_UNK_WORD = '[UNK]'
BERT_CLS_WORD = '[CLS]'
BERT_SEP_WORD = '[SEP]'

BERT_PAD = 0
BERT_UNK = 100
BERT_CLS = 101
BERT_SEP = 102
BERT_HIDDEN = 768

BERT_LAYERS = 12