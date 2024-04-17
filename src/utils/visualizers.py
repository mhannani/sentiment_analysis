import pandas as pd
import matplotlib.pyplot as plt


def visualize_class_frequencies(df: pd.DataFrame) -> None:
    """Visualize class frequencies as pie chart

    Args:
        df (pd.DataFrame): dataframe
    
    """
    
    # calculate the frequency of each unique value in the 'type' column
    type_counts = df['type'].value_counts()

    # create a pie chart using Matplotlib
    plt.figure(figsize=(4, 4))
    
    # create pie chart
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140)
    
    # set title
    plt.title('Distribution of Tweets Types')
    
    # equali axis dimentsion - circle pie chart
    plt.axis('equal')

    # show the pie chart
    plt.show()
