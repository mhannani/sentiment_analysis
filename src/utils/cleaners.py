import re
import emoji
import pandas as pd


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """clean the given DataFrame object and return the cleaned version.

    Basically:
        1. Removing missing values,
        2. Removing samples belonging to the mixed type,
        3. Dropping the column `class` as it's not relevent for the porject objective.
        4. Removing rows where the 'tweets' column contains empty strings.

    Args:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned version of the DataFrame.
    """
    
    # drop rows where tweets, type or class are NaN
    df.dropna(subset=['tweets', 'type', 'class'], inplace=True)
    
    # Renaming column to avoid reserved keywords conflicts
    df.rename(columns={'class': 'class_name'}, inplace=True)

    # clean tweets column
    df['tweets'] = df['tweets'].apply(clean_text)
    
    # Remove rows where the 'type' column has the value 'mixed'
    df = df[df['type'] != 'mixed']
    
    # Remove rows where 'tweets' column contains empty strings
    df = df[df['tweets'] != '']

    return df


def clean_text(text: str = None) -> str:
    """clean the given text and return the cleaned version

    Args:
        text (str, optional): text to clean. Defaults to None.

    Returns:
        str: Cleaned text
    """

    if text is None:
        return ""

    # remove <br> tag
    text = re.sub(r"<br>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    
    # Remove HTML anchor tags
    text = re.sub(r"<a\s+[^>]*>(.*?)</a>", "", text)
    
    # Remove special characters and punctuations
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)
    
    # Remove single characeters
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)

    # remove cammas
    text = text.replace(',', '')

    # Lowercase the text
    text = text.lower()

    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text)
    
    # Remove emojis and replace it with a shortcut
    text = emoji.demojize(text, language='en')
    
    # Trim leading and trailing spaces
    text = text.strip()

    return text


if __name__ == "__main__":
    
    text = "No comment ure the Best One ☝️ <br><br> <a href=http://www.youtube.com/results?search_query=%23Madd>#Madd</a> meZektini et en même temps tu as montré exactement ce qu’on vive au M9 lah ihsan 3wanhom Ga3 Barmo Hyathoum jwanat quand j’ai vu le clip sa m ‘a vraiment brisé le cœur d’avoir nos voisin dans cette état de drogue ... la vie sa devenu courte pr eux Sa me manque Derbna et welad homa Bezaff 😢😢😢😢 💔💔💔"
    
    print(text)
    print()
    cleaned_text = clean_text(text)
    
    print(cleaned_text)