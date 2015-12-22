
%% initialization

addpath(genpath('mill'));

train_data_file = '../data/data';
test_data_file = '../data/data';
model_file = 'model.txt';
kernel_type = 1;
global svm_folder;
svm_folder = 'mill/svm';

%% MILL

MIL_Run(['classify -t ' train_data_file ' -- train_only -m '...
        model_file ' -- inst_MI_SVM -Kernel ' num2str(kernel_type)]);
