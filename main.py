import argparse
from experiment.experiment import Experiment

def main():
    parser = argparse.ArgumentParser(description='Run experiments or models.')
    
    parser.add_argument('-m', '--mode', choices=['exp', 'model'], default='exp',
                        help='Specify whether to run an experiment or model.')
    
    parser.add_argument('-r', '--run_name', required=False,
                        help='Specify the run name for experiments.')
    
    args = parser.parse_args()

    if args.mode == 'exp':
        data_dir = "data/data_electronic/raw"
        exp = Experiment(data_dir)
        exp.run(run_name=args.run_name)
    elif args.mode == 'model':
        pass
        # Put model code here later
        print('run model')

if __name__ == "__main__":
    main()
