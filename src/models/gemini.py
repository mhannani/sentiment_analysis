import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompts.gemini import GeminiPrompt


class Gemini:
    """
    Google's Gemini/Bard as Sentiment analyzer.
    """
    
    def __init__(self, config: dict, google_api_key: str, gemini_prompt: GeminiPrompt):
        """Google's Gemini class constructor

        Args:
            config (dict): configuration object
            gemini_api_key (str): Gooel's Gemini api key
            gemini_prompt (GeminiPrompt) Gemini prompt constructor
        """
        
        # configuration object
        self.config = config
        
        # gemini api key
        self.google_api_key = google_api_key
        
        # gemini prompt constructor
        self.gemini_prompt = gemini_prompt

        # gemini model
        self.model = ChatGoogleGenerativeAI(model = "gemini-pro", google_api_key = self.google_api_key)

    def predict(self, request_sentence: str) -> dict:
        """Make prediction using Google's Gemini model.

        Args:
            request_sentence (str): Sentence to send to the Gemini model for inference.

        Returns:
            dict: Gemini response. e.g in the form {"review": request_sentence, "predicted_class": predicted_class}
        """
        
        # construct the gemini prompt
        gemini_prompt = self.gemini_prompt.create_prompt(request_sentence)
        
        # make prediction with self.model
        response = self.model.invoke(gemini_prompt)

        print(response)
