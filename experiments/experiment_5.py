import subprocess

num_epochs = 25
num_layers = 4
num_filters = 48
dim_reduction_types = ['no_dr', 'dilated_convolution']
do_types = ['1', '2', '3']

script_path = '../mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py'
script_template = 'python {0} --experiment_name {1} --num_epochs {2} --num_layers {3} --num_filters {4} --dim_reduction_type {5} --use_dropout={6} --use_gpu True'

for dm_type  in dim_reduction_types:
    for do_type in do_types:
        experiment_name = 'exp5_{0}_{1}'.format(dm_type, do_type)
        executable = script_template.format(script_path, experiment_name, num_epochs, num_layers, num_filters, dm_type, do_type)

        print('Starting: ' + executable)
        print()
        subprocess.run(executable, shell=True)
        print()
