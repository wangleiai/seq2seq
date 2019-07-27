
TEST = True

# 0:base model,
# 1:atten_model,
# 2:self_atten_model
MODEL_TYPE = 1

COSINEANNEALINGLR = False
T_MAX = 1000     #  学习率变化的周期，周期最大值时重新设置学习率。
ETA_MIN = 0.00001  # 使用COSINEANNEALINGLR时最小学习率
LR = 0.0001 # 如果COSINEANNEALINGLR=True，lr就是周期中最大的学习率，如果COSINEANNEALINGLR=False,就时普通的学习率

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
# dec_hid_dim = enc_hid_dim
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BID = True

BATCH_SIZE = 8
MIN_FREQ = 2
# DEVICE = 'cuda:0'
DEVICE = 'cpu:0'
SRC_MAX_SIZE = None
TRG_MAX_SIZE = None

LABEL_SMOOTH = False
LABEL_SMOOTHING = 0.1

MODEL_PATH = "models/model.pt"
TEST_SRC_PATH = ".data/multi30k/test2016.de"
TEST_TRG_PATH = ".data/multi30k/test2016.en"

