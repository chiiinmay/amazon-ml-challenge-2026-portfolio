import pandas as pd
from sklearn.model_selection import train_test_split


def create_validation_split(
    source1_path,
    validation_size=0.2,
    random_state=42
):
    source1 = pd.read_csv(
        source1_path,
        sep="\t"
    )

    entity_ids = source1["entity_id"].tolist()

    train_ids, val_ids = train_test_split(
        entity_ids,
        test_size=validation_size,
        random_state=random_state
    )

    return train_ids, val_ids


if __name__ == "__main__":

    train_ids, val_ids = create_validation_split(
        "dataset/train/train_source1.tsv"
    )

    print("Training Source 1 entities:", len(train_ids))
    print("Validation Source 1 entities:", len(val_ids))  