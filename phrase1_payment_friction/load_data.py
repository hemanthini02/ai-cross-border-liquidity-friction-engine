import pandas as pd

def load_settlement_data():
    # Load the settlement dataset
    df = pd.read_excel("data/dataset_2_cleaning_work.xlsx")

    print("Dataset loaded successfully")
    print("----------------------------------------")
    print("Shape (rows, columns):", df.shape)

    print("\nColumn names:")
    for col in df.columns:
        print(col)

    print("\nFirst 5 rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    load_settlement_data()
