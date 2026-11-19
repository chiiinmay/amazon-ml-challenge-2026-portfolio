import pandas as pd


def validate_matching_results(
    matching_path,
    test_source1_path,
    test_source2_path,
    test_source3_path
):
    errors = []

    # Read files
    matching = pd.read_csv(
        matching_path,
        sep="\t",
        keep_default_na=False
    )

    source1 = pd.read_csv(
        test_source1_path,
        sep="\t",
        keep_default_na=False
    )

    source2 = pd.read_csv(
        test_source2_path,
        sep="\t",
        keep_default_na=False
    )

    source3 = pd.read_csv(
        test_source3_path,
        sep="\t",
        keep_default_na=False
    )

    # Required columns
    required_columns = [
        "source1_entity_id",
        "matched_entity_ids"
    ]

    for column in required_columns:
        if column not in matching.columns:
            errors.append(
                f"Missing column: {column}"
            )

    if errors:
        return errors

    # Test Source 1 IDs
    test_source1_ids = set(
        source1["entity_id"]
    )

    result_source1_ids = set(
        matching["source1_entity_id"]
    )

    # Check missing Source 1 entities
    missing = test_source1_ids - result_source1_ids

    if missing:
        errors.append(
            f"Missing Source 1 entities: {len(missing)}"
        )

    # Check unexpected Source 1 entities
    unexpected = result_source1_ids - test_source1_ids

    if unexpected:
        errors.append(
            f"Unexpected Source 1 entities: {len(unexpected)}"
        )

    # Check duplicate Source 1 rows
    duplicates = matching[
        matching["source1_entity_id"].duplicated()
    ]

    if len(duplicates) > 0:
        errors.append(
            f"Duplicate Source 1 rows: {len(duplicates)}"
        )

    # Valid S2/S3 IDs
    valid_s2_ids = set(
        source2["entity_id"]
    )

    valid_s3_ids = set(
        source3["entity_id"]
    )

    valid_match_ids = valid_s2_ids | valid_s3_ids

    # Check each prediction
    for _, row in matching.iterrows():

        matched = row["matched_entity_ids"]

        if matched.strip() == "":
            continue

        ids = [
            x.strip()
            for x in matched.split(",")
            if x.strip()
        ]

        # Duplicate matches
        if len(ids) != len(set(ids)):
            errors.append(
                f"Duplicate match IDs for "
                f"{row['source1_entity_id']}"
            )

        # Invalid IDs
        invalid_ids = [
            x for x in ids
            if x not in valid_match_ids
        ]

        if invalid_ids:
            errors.append(
                f"Invalid match IDs for "
                f"{row['source1_entity_id']}: "
                f"{invalid_ids[:5]}"
            )

    return errors


if __name__ == "__main__":

    errors = validate_matching_results(
        "output/matching_results.tsv",
        "dataset/test/test_source1.tsv",
        "dataset/test/test_source2.tsv",
        "dataset/test/test_source3.tsv"
    )

    if errors:
        print("VALIDATION FAILED")

        for error in errors[:20]:
            print("-", error)

        print(
            f"\nTotal errors: {len(errors)}"
        )

    else:
        print("VALIDATION PASSED") 