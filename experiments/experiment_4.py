num_epochs = 25
num_layers = 4
num_filters = 64
dim_reduction_types = ['no_dr', 'strided_convolution', 'dilated_convolution', 'max_pooling', 'avg_pooling']

script_path = '../mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py'
script_template = 'python {0} --experiment_name {1} --num_epochs {2} --num_layers {3} --num_filters {4} --dim_reduction_type {5} --use_gpu True --use_dropout True'

for dm_type  in dim_reduction_types:
        experiment_name = 'exp4_{0}'.format(dm_type)
        executable = script_template.format(script_path, experiment_name, num_epochs, num_layers, num_filters, dm_type)

        print('Starting: ' + executable)
        print()
        subprocess.run(executable, shell=True)
        print()
