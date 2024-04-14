from langchain_core.messages import HumanMessage, SystemMessage
from typing import List


class GeminiPrompt:
    """Google's Gemini model Prompt"""
    
    def __init__(self, config: object, fuzzy_matches: List = []) -> None:
        """Gemini prompt.

        Args:
            config (object): configuration object
            fuzzy_matches (List, optional): Fuzzy matches examples. Defaults to [].
        """
        
        # configuration object
        self.config = config
        
        # fuzzy matches
        self.fuzzy_matches = fuzzy_matches
        
    def create_prompt(self, request_sentence: str) -> List:
        """_summary_

        Args:
            request_sentence (str): Comment or sentence to be sentimentally analyzed.

        Returns:
            List: List of messages to be fed to the model
        """
        
        # chat prompt template
        chat_messages = []
        
        # system message
        system_message = self.config["prompting"]["gemini_system_message"]
        
        # constructing chat message
        # adding system message
        chat_messages.append(SystemMessage(content = system_message))
        
        # adding human message
        chat_messages.append(HumanMessage(content = request_sentence))

        return chat_messages
