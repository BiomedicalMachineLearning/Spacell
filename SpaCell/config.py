

# path config
META_PATH = '/Users/xiao/BIOINFOR/Project/spacell/dataset/metadata/mouse_sample_names_sra.tsv'
IMG_PATH='/Users/xiao/BIOINFOR/Project/spacell/dataset/image/'
CM_PATH='/Users/xiao/BIOINFOR/Project/spacell/dataset/cm/'
ATM_PATH=None
TILE_PATH='/Users/xiao/BIOINFOR/Project/spacell/dataset/tile/'
DATASET_PATH='/Users/xiao/BIOINFOR/Project/spacell/dataset/'
TEMPLATE_IMG = '/Users/xiao/BIOINFOR/Project/spacell/dataset/image/CN63_C1_HE.jpg'

# image config
SIZE=299, 299

NORM_METHOD = 'vahadane'
# count matrix config

THRESHOLD_GENE = 0.01
THRESHOLD_SPOT = 0.01
MIN_EXP = 1

# reproducibility
seed = 37

# color_map 
color_map = ['#ff8aff', '#6fc23f', '#af63ff', '#eaed00', '#f02449', '#00dbeb', '#d19158', '#9eaada', '#89af7c', '#514036']

# model config
n_classes = 4
batch_size = 32