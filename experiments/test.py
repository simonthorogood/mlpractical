import subprocess

script_path = '../mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py'
script_template = 'python {0} --experiment_name {1} --num_epochs {2} --num_layers {3} --num_filters {4} --dim_reduction_type {5} --use_gpu True'
executable = script_template.format(script_path, 'test', 1, 2, 8, 'max_pooling')

subprocess.run(executable, shell=True)