# Test name
test_name = 'name'
sampling_option = 'oversampling'

# Batch size
train_batch_size = 16
val_batch_size = 16
test_batch_size = 1

# Num_classes
num_classes = 4

# Model
model = 'resnet'

# Feature extract or train all parameters
feature_extract = False

# Num epoch
epoch_max = 100

# Folder image dir
data_dir = './data/'

# Save dir
save_dir = '../checkpoint' + '/' + test_name

# Test epoch: what epoch to load
inference_epoch = 2

# Resume training
continue_epoch = 0

# Early stopping
patience = 50

# Regularization
lambda_l1 = 1e-5

# Resize image
resize = 1024

# Repeat for inference
repeat = 100

# Learning rate
lr = 1e-5

# Patch size
img_size = 224

# ResNet layers
resnet = 'resnet18'
