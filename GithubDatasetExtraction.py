import pandas as pd

def get_article(id):
    # Load the CSV
    df = pd.read_csv('en_train_subtask_1.csv')
    row_index = id

    # Extract row
    specific_row = df.iloc[row_index]

    # Return article
    return specific_row.to_dict()["articles"]