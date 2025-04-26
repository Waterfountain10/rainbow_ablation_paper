import pandas as pd

# --- Configuration ---
input_csv_file = "./bad_format/USDCHF_H4.csv"  # Replace with your input file name
output_csv_file = "./CHFUSD_H4.csv"  # Replace with your desired output file name

# --- Column Names (Adjust if yours are different) ---
# Make sure these match the exact names in your CSV header
date_col = "Time"
open_col = "Open"
high_col = "High"
low_col = "Low"
close_col = "Close"
adj_close_col = "Adj Close"  # Set to None if not present
volume_col = "Volume"  # Set to None if not present

try:
    # Read the CSV file
    df = pd.read_csv(input_csv_file)

    # --- Perform the Conversion ---
    # Store original High/Low temporarily because we need them for the swap
    original_high = df[high_col].copy()
    original_low = df[low_col].copy()

    # Calculate new values (handle potential division by zero if necessary, though unlikely for FX)
    df[open_col] = 1 / df[open_col]
    df[high_col] = 1 / original_low  # New High = 1 / Old Low
    df[low_col] = 1 / original_high  # New Low = 1 / Old High
    df[close_col] = 1 / df[close_col]

    if adj_close_col and adj_close_col in df.columns:
        df[adj_close_col] = 1 / df[adj_close_col]

    # --- Optional: Round to a reasonable number of decimal places ---
    # FX rates often use 4-6 decimal places
    price_cols = [open_col, high_col, low_col, close_col]
    if adj_close_col and adj_close_col in df.columns:
        price_cols.append(adj_close_col)
    df[price_cols] = df[price_cols].round(6)  # Adjust precision as needed

    # --- Reorder columns if desired (optional) ---
    # Example: Keep original order
    cols_to_keep = [date_col, open_col, high_col, low_col, close_col]
    if adj_close_col and adj_close_col in df.columns:
        cols_to_keep.append(adj_close_col)
    if volume_col and volume_col in df.columns:
        cols_to_keep.append(volume_col)

    # Ensure all original columns intended to be kept are included
    # Add any other columns from the original df that should be preserved
    other_cols = [col for col in df.columns if col not in cols_to_keep]
    final_cols = cols_to_keep + other_cols
    df = df[final_cols]

    # --- Save the result to a new CSV ---
    df.to_csv(output_csv_file, index=False)  # index=False prevents writing row numbers

    print(f"Successfully converted data and saved to '{output_csv_file}'")

except FileNotFoundError:
    print(f"Error: Input file '{input_csv_file}' not found.")
except KeyError as e:
    print(f"Error: Column '{e}' not found in the CSV. Please check column names.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
