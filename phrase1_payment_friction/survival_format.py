import pandas as pd

def create_survival_dataset():
    # Load dataset
    df = pd.read_excel("data/dataset_2_cleaning_work.xlsx")

    # Convert settlement delay to numeric
    df["settlement_delay_minutes"] = pd.to_numeric(
        df["settlement_delay_minutes"], errors="coerce"
    )

    # Drop rows where delay could not be converted
    df = df.dropna(subset=["settlement_delay_minutes"])

    # Keep only positive delays
    df = df[df["settlement_delay_minutes"] > 0]

    # Create event column (1 = settlement completed)
    df["event"] = 1

    # Select columns needed for survival analysis
    survival_df = df[
        [
            "settlement_delay_minutes",
            "event",
            "corridor",
            "firm",
            "firm_type",
            "payment instrument",
            "access point",
            "receiving network coverage",
            "pickup method",
            "transparent",
            "cc1 fx margin",
            "cc1 lcu fee",
            "cc1 total cost %"
        ]
    ]

    print("Survival dataset created")
    print("----------------------------------------")
    print("Shape:", survival_df.shape)

    print("\nFirst 5 rows:")
    print(survival_df.head())

    print("\nEvent distribution:")
    print(survival_df["event"].value_counts())

    print("\nSettlement delay summary:")
    print(survival_df["settlement_delay_minutes"].describe())

    return survival_df


if __name__ == "__main__":
    create_survival_dataset()
