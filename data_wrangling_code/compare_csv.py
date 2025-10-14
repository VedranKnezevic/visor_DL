#!/usr/bin/env python3
import argparse
import pandas as pd

def compare_dataframes(df1, df2):
    """Compare two DataFrames and return rows missing in each."""
    # Find rows in df1 not in df2
    missing_in_df2 = df1.merge(df2.drop_duplicates(), how='left', indicator=True)
    missing_in_df2 = missing_in_df2[missing_in_df2['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Find rows in df2 not in df1
    missing_in_df1 = df2.merge(df1.drop_duplicates(), how='left', indicator=True)
    missing_in_df1 = missing_in_df1[missing_in_df1['_merge'] == 'left_only'].drop(columns=['_merge'])

    return missing_in_df2, missing_in_df1


def main():
    parser = argparse.ArgumentParser(
        description="Compare two CSV files and show which rows are missing in each."
    )
    parser.add_argument("csv1", help="Path to the first CSV file")
    parser.add_argument("csv2", help="Path to the second CSV file")
    parser.add_argument("--output", "-o", help="Optional prefix for output CSV files", default=None)

    args = parser.parse_args()

    # Load CSV files
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    # Compare DataFrames
    missing_in_df2, missing_in_df1 = compare_dataframes(df1, df2)

    # Print summary
    print(f"Rows in {args.csv1} missing in {args.csv2}: {len(missing_in_df2)}")
    print(f"Rows in {args.csv2} missing in {args.csv1}: {len(missing_in_df1)}")

    # Optionally save results
    if args.output:
        out1 = f"{args.output}_missing_in_{args.csv2.split('/')[-1]}"
        out2 = f"{args.output}_missing_in_{args.csv1.split('/')[-1]}"
        missing_in_df2.to_csv(out1, index=False)
        missing_in_df1.to_csv(out2, index=False)
        print(f"Results saved to:\n  {out1}\n  {out2}")
    else:
        print("\n--- Rows missing in first file ---")
        print(missing_in_df2)
        print("\n--- Rows missing in second file ---")
        print(missing_in_df1)


if __name__ == "__main__":
    main()
