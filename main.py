from experiment.experiment import Experiment

# Get the directory of the current script (main.py)
data_dir = "data/data_electronic/raw_test"

exp = Experiment(data_dir)
exp.run()

# train_model()
