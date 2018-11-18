import subprocess

num_epochs = 25
num_layer_settings = [2, 3, 4, 5]

num_filters_settings = [16, 32, 64, 128]
dim_reduction_types = {
    'max_pooling': 96,
    'avg_pooling': 128,
    'strided_convolution': 96,
    'dilated_convolution': 128
}


script_path = '../mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py'
script_template = 'python {0} --experiment_name {1} --num_epochs {2} --num_layers {3} --num_filters {4} --dim_reduction_type {5} --use_gpu True'

for dm_type, num_filters  in dim_reduction_types.items():
    for num_layers in num_layer_settings:
        experiment_name = 'exp2_{0}_{1}'.format(dm_type, num_layers)
        executable = script_template.format(script_path, experiment_name, num_epochs, num_layers, num_filters, dm_type)

        print('Starting: ' + executable)
        print()
        subprocess.run(executable, shell=True)
        print()
