import pandas as pd


def load_ground_truth(path):
    df = pd.read_csv(
        path,
        sep="\t",
        keep_default_na=False
    )

    ground_truth = {}

    for _, row in df.iterrows():
        source1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        if matched.strip() == "":
            matched_ids = []
        else:
            matched_ids = [
                x.strip()
                for x in matched.split(",")
                if x.strip()
            ]

        ground_truth[source1_id] = matched_ids

    return ground_truth


if __name__ == "__main__":

    path = "dataset/train/train_ground_truth.tsv"

    ground_truth = load_ground_truth(path)

    print("Total Source 1 entities:", len(ground_truth))

    # Show first 5 examples
    for source1_id, matched_ids in list(ground_truth.items())[:5]:
        print(
            source1_id,
            "->",
            matched_ids
        ) 