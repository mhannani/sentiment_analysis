import fasttext

class FastTextEmbeddings:
    """Wrapper class for FastText embeddings"""

    def __init__(self, model_path: str) -> None:
        """Initialize FastText model

        Args:
            model_path (str): Path to the FastText model file
        """
        self.model = fasttext.load_model(model_path)

    def get_sentence_vector(self, text: str) -> list:
        """Get the FastText embedding vector for a sentence

        Args:
            text (str): Input sentence

        Returns:
            list: FastText embedding vector for the sentence
        """
        return self.model.get_sentence_vector(text)