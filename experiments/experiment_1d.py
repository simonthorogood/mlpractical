import subprocess

num_epochs = 75
num_layers = 4
num_filters_settings = [16, 32, 48, 64, 96]
dim_reduction_types = ['no_dr']

script_path = '../mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py'
script_template = 'python {0} --experiment_name {1} --num_epochs {2} --num_layers {3} --num_filters {4} --dim_reduction_type {5} --use_gpu True'

for dm_type  in dim_reduction_types:
    for num_filters in num_filters_settings:
        experiment_name = 'exp1d_{0}_{1}'.format(dm_type, num_filters)
        executable = script_template.format(script_path, experiment_name, num_epochs, num_layers, num_filters, dm_type)

        print('Starting: ' + executable)
        print()
        subprocess.run(executable, shell=True)
        print()
