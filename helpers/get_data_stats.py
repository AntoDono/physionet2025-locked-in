import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_pipeline import load_data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, required=True)
    return parser

def run(args):
    df = load_data(args.data_folder, use_cache=False)
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run(args)