import pandas as pd


def main():
    df = pd.read_csv("SampleData.csv")

    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()