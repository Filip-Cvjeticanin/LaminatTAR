import pandas as pd

# Load the CSV
df = pd.read_csv('en_train_subtask_1.csv')

# --- OPTION A: Query by Row Index (Position) ---
# Example: Get the 5th row (remember Python starts at 0, so index 4 is the 5th row)
row_index = 0
specific_row = df.iloc[row_index]

# --- OPTION B: Query by a Unique Value (ID/Name) ---
# Example: Get the row where the 'ID' column is 101
# specific_row = df[df['ID'] == 101].iloc[0]

print(f"Details for row {row_index}:")
print(specific_row)

# Accessing a specific cell from that row
# print(f"The name in this row is: {specific_row['Name']}")