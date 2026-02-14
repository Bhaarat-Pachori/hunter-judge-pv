import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
import os

def split_cadec_data(data_path, output_dir, val_size=0.1, test_size=0.2, random_state=42):
    """
    Splits the CADEC v2 dataset into training, validation, and test sets using 
    GroupShuffleSplit on the 'source_file' column to prevent data leakage,
    followed by a standard train_test_split for the training set into training and validation.

    Args:
        data_path (str): Path to the cadec_v2.csv file.
        output_dir (str): Directory to save the train.csv and test.csv files.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
    """

    # 1. Load the dataset
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file {data_path} was not found.")
        return

    # 2. Prepare the data for GroupShuffleSplit
    groups = df['source_file']

    # 3. Initialize and perform the split
    group_shuffle_split = GroupShuffleSplit(test_size=test_size, random_state=random_state)
    # train_idx, test_idx = next(group_shuffle_split.split(df, groups=groups))
    train_val_idx, test_idx = next(group_shuffle_split.split(df, groups=groups))

    # train_df = df.iloc[train_idx]
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # Split the training data into training and validation sets
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, 
                                            random_state=random_state, stratify=train_val_df['label'])


    # 4. Verify no leakage
    train_files = set(train_df['source_file'])
    test_files = set(test_df['source_file'])
    intersection = train_files.intersection(test_files)
    assert len(intersection) == 0, "Leakage detected: source_file values present in both train and test sets."

    # 5. Class balance check
    val_positive_percent = val_df['label'].value_counts(normalize=True).get(1, 0) * 100
    train_positive_percent = train_df['label'].value_counts(normalize=True).get(1, 0) * 100
    test_positive_percent = test_df['label'].value_counts(normalize=True).get(1, 0) * 100

    # 6. Create output directory if it doesn't exist
    test_output_dir = os.path.join(os.path.dirname(output_dir), "test_split")
    os.makedirs(test_output_dir, exist_ok=True)

    # 7. Save the dataframes to CSV
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(test_output_dir, 'test.csv'), index=False)

    # 8. Print the statistics
    print("Data split complete.")
    print(f"Train set size: {len(train_df)} sentences")
    print(f"Validation set size: {len(val_df)} sentences")
    print(f"Test set size: {len(test_df)} sentences")
    print(f"Positive label percentage in validation set: {val_positive_percent:.2f}%")
    print(f"Positive label percentage in train set: {train_positive_percent:.2f}%")
    print(f"Positive label percentage in test set: {test_positive_percent:.2f}%")
    print("Leakage check passed: No source_file values are present in both train and test sets.")


if __name__ == "__main__":
    data_path = "hunter-judge-pv/data/cadec_v2/cadec_v2.csv"
    output_dir = "hunter-judge-pv/data/cadec_v2/data_splits"

    # Run the splitting function
    split_cadec_data(data_path, output_dir)

