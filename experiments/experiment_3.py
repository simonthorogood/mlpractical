num_epochs = 25
num_layers = 4
num_filters = 64
dim_reduction_types = ['no_dr', 'strided_convolution', 'dilated_convolution', 'max_pooling', 'avg_pooling']
wd_vals = [1e-05, 1e-04, 1e-03]

script_path = '../mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py'
script_template = 'python {0} --experiment_name {1} --num_epochs {2} --num_layers {3} --num_filters {4} --dim_reduction_type {5} --weight_decay_coefficient {6} --use_gpu True'

for dm_type  in dim_reduction_types:
    for wd_val in wd_vals:
        experiment_name = 'exp3_{0}_{1}'.format(dm_type, wd_val)
        executable = script_template.format(script_path, experiment_name, num_epochs, num_layers, num_filters, dm_type, wd_val)

        print('Starting: ' + executable)
        print()
        subprocess.run(executable, shell=True)
        print()
