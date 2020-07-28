
mode = 'metrics' #'inference'#'train' #

data_folder = 'facnp8'
data_path = '../cnp_data/'

preproc_path = '../cnp_data/facnp860_cv5'#'../cnp_data/cnpdata6b_cv5'#''#'../cnp_data/facnp860_cv5'#''#'../cnp_data/cnpdataX6_split_eq'#../cnp_data/cnpdataX6_split_clahe'#'../cnp_data/cnpdataX1_split'#'../cnp_data/cnpdata_split'#'../cnp_data/cnpdataX1twice_split'#

# premodel_fdr = './out/out0/premodel' #'' #

inf_model_folder = r'/raid/jamseth/root_dev/exp_cnptool/pych_cnptool_baseline/pych_cnptool_f4/out/out4/Model'
inf_out_folder = r'/raid/jamseth/root_dev/exp_cnptool/pych_cnptool_baseline/pych_cnptool_f4/out/out4/auto_rev'
inf_cv_folder = '../cnp_data/cnpdata6b_cv5/fold4/Done' # to know images in fold
inf_data_folder = r'../cnp_data/facnp8/Done'

##
# Metrics parameters
pred_folder = r'/raid/jamseth/root_dev/exp_cnptool/pych_cnptool_baseline'#/pych_cnptool_f0/out/out0/auto_rev/ch1'
gt_folder = r'../cnp_data/facnp8/NP'
weights_folder = r'../cnp_data/facnp8/Valid'
cv_folder = '../cnp_data/cnpdata6b_cv5'#/fold0/Done' # to know images in fold
##

kcrossval = 5
nfold = 4 # TODO: to be removed

visualization_step = 10000000
valid_visual_step = 10000000
test_visual_step = 10000000
save_results = True#False# # check if still actually used
verbose = False#True#

gpu_list = '3'

crop_height = 1400#2000
crop_width = 1400#2000

resize_height = 448#896#224#
resize_width = 448#896#448#896#448#224#

preproc_mode = ''#''eq'#'clahe'#'local_avg'#

train_fraction = 0.8#0.85#1#0.98#0.70#

# data_augm_rate = 0 # if 0 then no data augmentation enabled

nclasses = 2
input_mode = 'cnp'#'vld'#
mask_mode = 'vld'#''#

activation = 'softmax'#'sigmoid'#
network = 'unet' #'iunet' #
loss = 'gdl' #'iterative'#
# activation = 'softmax'#
# network = 'unet' #
# loss = 'gdl' #

# Training
train_batch_size = 4
valid_batch_size = 12
# test_batch_size = 2 # hardwired to test_nelem
num_epochs = 50001

# Model saving
models_to_keep = 5
model_save_step = 1000

train_augm = True#False#
train_shuffle = True#False#
valid_shuffle = False#True#
test_shuffle = False#True#

optimizer = 'sgd'#'adam'#

batch_norm = True#False
start_learning_rate = 0.005#0.001
learning_decay_step = 100000
learning_decay_rate = 0.9
momentum = 0.9
staircase = False#True

# l2_beta = 0.0005
dropout_keep = 0.5

# Output
output_path = 'out'
model_folder = 'Model'
model_name = 'model'
summary_folder = 'Summary'
train_sm_folder = 'Train'
valid_sm_folder = 'Valid'
test_sm_folder = 'Test'
results_folder = 'results'

# Validation
# train_fraction = 0.98#0.70#0.85

graph_file_name = 'output_graph.pb'








