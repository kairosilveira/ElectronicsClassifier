import argparse
from experiment.experiment import Experiment
from app.camera_app import run

def main():
    parser = argparse.ArgumentParser(description='Run experiments or app.')
    
    parser.add_argument('-m', '--mode', choices=['exp', 'app'], default='app',
                        help='Specify whether to run an experiment or app.')
    
    parser.add_argument('-r', '--run_name', required=False,
                        help='Specify the run name for experiments.')
    
    args = parser.parse_args()

    if args.mode == 'exp':
        data_dir = "data/data_electronic/raw"
        exp = Experiment(data_dir)
        exp.run(run_name=args.run_name)
    elif args.mode == 'app':
        run()

if __name__ == "__main__":
    main()
