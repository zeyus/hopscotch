import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging


def load_mocap_data(path: Path) -> pd.DataFrame:
    """
    Load mocap data from a given path and return a DataFrame with subject, condition, and session.

    Args:
        path (Path): Path to the mocap data.

    Returns:
        DataLoader: DataLoader for the mocap data.
    """

    # try and load cached feather file
    try:
        dataset = pd.read_feather(path / "mocap_data.feather")
        logging.info("Loaded cached mocap data.")
        return dataset
    except FileNotFoundError:
        logging.info("Cached mocap data not found. Loading from source.")

    COL_REF = set([
        "Frame",
        "Time",
        "head.X",
        "head.Y",
        "head.Z",
        "foot_front_r.X",
        "foot_front_r.Y",
        "foot_front_r.Z",
        "foot_back_r.X",
        "foot_back_r.Y",
        "foot_back_r.Z",
        "knee_under_r.X",
        "knee_under_r.Y",
        "knee_under_r.Z",
        "knee_over_r.X",
        "knee_over_r.Y",
        "knee_over_r.Z",
        "wrist_r.X",
        "wrist_r.Y",
        "wrist_r.Z",
        "elbow_r.X",
        "elbow_r.Y",
        "elbow_r.Z",
        "shoulder_r.X",
        "shoulder_r.Y",
        "shoulder_r.Z",
        "hip_front_r.X",
        "hip_front_r.Y",
        "hip_front_r.Z",
        "hip_back_r.X",
        "hip_back_r.Y",
        "hip_back_r.Z",
        "foot_back_l.X",
        "foot_back_l.Y",
        "foot_back_l.Z",
        "foot_front_l.X",
        "foot_front_l.Y",
        "foot_front_l.Z",
        "knee_under_l.X",
        "knee_under_l.Y",
        "knee_under_l.Z",
        "knee_over_l.X",
        "knee_over_l.Y",
        "knee_over_l.Z",
        "hip_front_l.X",
        "hip_front_l.Y",
        "hip_front_l.Z",
        "hip_back_l.X",
        "hip_back_l.Y",
        "hip_back_l.Z",
        "wrist_l.X",
        "wrist_l.Y",
        "wrist_l.Z",
        "elbow_l.X",
        "elbow_l.Y",
        "elbow_l.Z",
        "shoulder_l.X",
        "shoulder_l.Y",
        "shoulder_l.Z",
        "floor_start_l.X",
        "floor_start_l.Y",
        "floor_start_l.Z",
        "floor_start_r.X",
        "floor_start_r.Y",
        "floor_start_r.Z",
        "floor_end_r.X",
        "floor_end_r.Y",
        "floor_end_r.Z",
        "floor_end_l.X",
        "floor_end_l.Y",
        "floor_end_l.Z",
    ])
    files_exclude = [
        "26_k1",
        "27_h1",
        "32_k1",
    ]
    files = list(path.glob("*.tsv"))
    files.sort(key=lambda x: int(x.stem.split("_")[0]))
    expected_columns = 71
    dataset = None
    for file in tqdm(files, desc="Loading mocap data", unit="tsvs"):
        # Skip files that are in the exclude list
        if file.stem in files_exclude:
            logging.warning(f"Skipping file {file} because it is in the exclude list.")
            continue
        # trim any leading and traling whitespace from the file's first line
        with open(file, "r+") as f:
            line = f.readlines()
            f.seek(0)
            f.writelines(
                [
                    line[0].strip().replace(" ", ".") + "\n"
                ] + line[1:]
            )

        # Read the file
        df = pd.read_csv(file, sep="\t", low_memory=False, skip_blank_lines=True)
        # Check the column length is the expected length
        if df.shape[1] < expected_columns:
            # find the differing columns
            diff = set(COL_REF).difference(df.columns.tolist())
            logging.warning(f"File {file} has {df.shape[1]} columns, expected {expected_columns}.")
            logging.warning(f"Missing columns: {diff}")
        elif df.shape[1] > expected_columns:
            # find the differing columns
            logging.warning(f"File {file} has {df.shape[1]} columns, expected {expected_columns}.")
            # if there's a column "X" drop it
            if "X" in df.columns:
                df.drop(columns=["X"], inplace=True)
                logging.warning(f"Dropping column X from {file}.")

        # Extract the subject, condition, and the obstacle from the filename
        filename = file.stem
        subject, rest = filename.split("_")
        # Add the subject, condition, and session to the DataFrame
        df["subject"] = int(subject)
        df["condition"] = rest[0]
        df["obstacles"] = int(rest[1])
        # add file name to the DataFrame
        df["filename"] = filename
        # Append the DataFrame to the dataset
        if dataset is None:
            dataset = df
        else:
            dataset = pd.concat([dataset, df], ignore_index=True)
        # save the dataset to a file
        dataset.to_feather("data/mocap_data.feather")
        logging.info("Saved mocap data to data/mocap_data.feather.")
        # save csv as well
        dataset.to_csv("data/mocap_data.csv")
        logging.info("Saved mocap data to data/mocap_data.csv.")
        
    return dataset
