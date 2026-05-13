import pandas as pd

def get_number_of_articles(articles) -> int:
    """
    Returns the number of articles present in the articles dataframe.
    :param articles:
    :return:
    """
    pass

def get_article(ID: int) -> str:
    """
    Returns the article with the given id.
    :param ID: 
    :return: 
    """
    # Load the CSV
    df = pd.read_csv('en_train_subtask_1.csv')
    row_index = ID

    # Extract row
    specific_row = df.iloc[row_index]

    # Return article
    return specific_row.to_dict()["articles"]