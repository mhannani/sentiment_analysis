import re


class Preprocessor:
    """
    Reviews or comments preprocessor
    """

    def __init__(self):
        """
        Class constructor
        """
    
    def clean_text(self, text: str = None) -> str:
        """
        Clean the provided text
        """

        if text is None:
            return ""

        # Remove special characters and punctuations
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove single characeters
        text = re.sub(r"\b[a-zA-Z]\b", " ", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]*>", " ", text)

        # Lowercase the text
        text = text.lower()

        # Remove extra whitespaces
        text = re.sub(r"\s+", " ", text)

        # Trim leading and trailing spaces
        text = text.strip()

        return text
