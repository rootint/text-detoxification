import pandas as pd


def preprocess(path="data/raw/filtered.tsv"):
    df = pd.read_csv(path, sep="\t")

    # Cleaning data
    print("Cleaning data...")
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df.rename(columns={"lenght_diff": "length_diff"}, inplace=True)

    # Lowercasing and removing special chars
    print("Lowercasing and removing special chars...")
    df["reference"] = (
        df["reference"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.strip()
    )
    df["translation"] = (
        df["translation"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.strip()
    )

    # Dropping unnecessary columns
    print("Dropping unnecessary columns...")
    filtered_data = df.drop(["similarity", "length_diff"], axis=1)

    # Swapping reference and translation where needed
    print("Swapping reference and translation where needed...")
    filtered_data_swapped = filtered_data.copy()
    needs_swap = filtered_data_swapped["ref_tox"] < filtered_data_swapped["trn_tox"]
    filtered_data_swapped.loc[
        needs_swap, ["reference", "translation"]
    ] = filtered_data_swapped.loc[needs_swap, ["translation", "reference"]].values
    filtered_data_swapped.loc[
        needs_swap, ["ref_tox", "trn_tox"]
    ] = filtered_data_swapped.loc[needs_swap, ["trn_tox", "ref_tox"]].values

    # Dropping duplicates
    print("Dropping duplicates...")
    filtered_data_swapped = filtered_data_swapped.drop_duplicates()

    # Keeping rows with high toxicity threshold
    print("Keeping rows with high toxicity threshold...")
    high_toxicity_threshold = 0.7
    filtered_data_diff = filtered_data_swapped.copy()
    filtered_data_diff["tox_diff"] = (
        filtered_data_diff["ref_tox"] - filtered_data_diff["trn_tox"]
    ).abs()
    final_data = filtered_data_diff[
        filtered_data_diff["tox_diff"] > high_toxicity_threshold
    ].sort_values(by="ref_tox", ascending=True)

    # Dropping rows with more that have strings longer than 256 chars
    print("Dropping rows with more that have strings longer than 256 chars...")
    final_data["reference_length"] = final_data["reference"].apply(len)
    final_data["translation_length"] = final_data["translation"].apply(len)
    data = final_data[
        (final_data["reference_length"] <= 256)
        & (final_data["translation_length"] <= 256)
    ]
    training_data = data.drop(
        ["reference_length", "translation_length", "tox_diff", "ref_tox", "trn_tox"],
        axis=1,
    )

    # Saving and removing NaNs
    print("Saving and removing NaNs...")
    training_data.to_csv("src/data/training_data.csv", index=False)
    dff = pd.read_csv(
        "src/data/training_data.csv",
        dtype={"toxic_column": str, "nontoxic_column": str},
    )
    dff.dropna(inplace=True)
    dff.to_csv("src/data/training_data.csv", index=False)
    print("Done!")


if __name__ == "__main__":
    preprocess()
