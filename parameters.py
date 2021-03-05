
data_folder = 'facnp8'
data_path = '../exp_cnptool/exp_cnpdata/'

preproc_path = '../exp_cnptool/cnp_data/cnpdata7_cv5'

gpu_list = '1'

network = 'unet' #'iunet' #
train_batch_size = 4
valid_batch_size = 12
num_epochs = 50001

kcrossval = 5
nfold = 4 # None for full cross-validation

# Format data
crop_height = 1400#2000
crop_width = 1400#2000
resize_height = 448#896#224#
resize_width = 448#896#448#896#448#224#
preproc_mode = ''#''eq'#'clahe'#'local_avg'#
train_fraction = 0.8#0.85#1#0.98#0.70#

# Training
train_augm = True
train_shuffle = True
valid_shuffle = False
test_shuffle = False

nclasses = 2
input_mode = 'cnp'#'vld'#
mask_mode = 'vld'#''#

activation = 'softmax'
loss = 'gdl' #'iterative'#
optimizer = 'sgd'#'adam'#

batch_norm = True
start_learning_rate = 0.005
learning_decay_step = 100000
learning_decay_rate = 0.9
momentum = 0.9
staircase = False#True

# l2_beta = 0.0005
dropout_keep = 0.5

verbose = False#True#

# Output
output_path = 'out'
model_folder = 'Model'
model_name = 'model'
summary_folder = 'Summary'
train_sm_folder = 'Train'
valid_sm_folder = 'Valid'
test_sm_folder = 'Test'
results_folder = 'results'

# Model saving
models_to_keep = 5
model_save_step = 1000

##################################################
# inference
inf_model_folder = r'./models/out1'
inf_out_folder = r'./out/out1'
inf_cv_folder = '../exp_cnptool/cnp_data/cnpdata6b_cv5/fold1/Done'  # to know images in fold
inf_data_folder = r'../exp_cnptool/cnp_data/facnp8/Done'

# inf_model_folder = r'/raid/jamseth/root_dev/exp_cnptool/pych_cnptool_baseline/pych_cnptool_f4/out/out4/Model'
# inf_out_folder = r'/raid/jamseth/root_dev/exp_cnptool/pych_cnptool_baseline/pych_cnptool_f4/out/out4/auto_rev'
# inf_cv_folder = '../cnp_data/cnpdata6b_cv5/fold4/Done' # to know images in fold
# inf_data_folder = r'../cnp_data/facnp8/Done'

##################################################
# metrics
pred_folder = r'/raid/jamseth/root_dev/exp_cnptool/pych_cnptool_baseline'#/pych_cnptool_f0/out/out0/auto_rev/ch1'
gt_folder = r'../cnp_data/facnp8/NP'
weights_folder = r'../cnp_data/facnp8/Valid'
cv_folder = '../cnp_data/cnpdata6b_cv5'#/fold0/Done' # to know images in fold










