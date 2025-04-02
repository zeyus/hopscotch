from processing.loader import load_mocap_data
from pathlib import Path
import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Hello World")
    parser.add_argument(
        "--destructive",
        action="store_true",
        help="Run the script in destructive mode.",
    )
    return parser.parse_args()


def main(destructive: bool = False):
    if not destructive:
        logging.info("This script will probably end up altering your data.")
        logging.info("Please run with --destructive to confirm.")
        return
    else:
        logging.warning("Running in destructive mode.")     
    logging.info("Loading mocap data...")
    dataset = load_mocap_data(Path("data/"))
    logging.info("Loaded mocap data.")
    logging.info("Data shape: %s", dataset.shape)
    

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    destructive = args.destructive
    main(destructive)
