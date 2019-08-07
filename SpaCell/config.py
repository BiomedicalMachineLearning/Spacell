

# path config
META_PATH = '/Users/xiao.tan/Downloads/dataset/metadata/mouse_sample_names_sra.tsv'
IMG_PATH = '/Users/xiao.tan/Downloads/dataset/image/'
CM_PATH = '/Users/xiao.tan/Downloads/dataset/cm/'
ATM_PATH = None
TILE_PATH = '/Users/xiao.tan/Downloads/dataset/tile/'
DATASET_PATH = '/Users/xiao.tan/Downloads/dataset/'
TEMPLATE_IMG = '/Users/xiao.tan/Downloads/dataset/image/CN63_C1_HE.jpg'

# image config
SIZE = 299, 299
N_CHANNEL = 3

NORM_METHOD = 'vahadane'
# count matrix config

THRESHOLD_GENE = 0.01
THRESHOLD_SPOT = 0.01
MIN_EXP = 1

# metadata config
SAMPLE_COLUMN = 'sample_name'
LABEL_COLUMN = 'age'
CONDITION_COLUMN = 'breed'
CONDITION = 'B6SJLSOD1-G93A'
ADDITIONAL_COLUMN = 2 if CONDITION_COLUMN else 1

# reproducibility
seed = 37

# color_map 
color_map = ['#ff8aff', '#6fc23f', '#af63ff', '#eaed00', '#f02449', '#00dbeb', '#d19158', '#9eaada', '#89af7c', '#514036']

# classification model config
n_classes = 4
batch_size = 32
epochs = 10
train_ratio = 0.7
