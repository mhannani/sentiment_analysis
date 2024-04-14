import openai
from typing import List
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential
from langchain.schema.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_tools import JsonOutputToolsParser

from src.prompts.chatgpt import GPTPrompt
from src.types.review_class import ReviewClass

class GPT:
    """
    GPT as a Sentiment Analyzer model
    """

    def __init__(self, config, openai_api_key: str, gpt_prompt: GPTPrompt, model_name: str = 'gpt-3.5-turbo', temperature: float = 0) -> None:
        """
        Class constructor for GPT model as API.

        :param config
            Configuration object
        :param openai_api_key str
            OpenAI API key
        :param gpt_prompt GPTPrompt
            The GPT prompt
        :param model str
            Model name
        :param temperature float
            Temperature
        :param messages List
            Messages to pass to the model

        return None
        """

        # Configuration object
        self.config = config

        # Openai_api_key
        self.openai_api_key: str = openai_api_key

        # GPT prompt
        self.gpt_prompt: GPTPrompt = gpt_prompt

        # Model name
        self.model_name: str = model_name

        # Temperator
        self.temperature: float = temperature

        # Setting openai api key
        openai.api_key = self.openai_api_key

        # Instantiate the chatOpenAI class
        self.model = ChatOpenAI(model_name = self.model_name, temperature = self.temperature, openai_api_key = self.openai_api_key).bind_tools([ReviewClass])

        # Output Parser
        self.output_parser = JsonOutputToolsParser(key_name="ReviewClass", first_tool_only=True)

    def predict(self, request_sentence: str) -> AIMessage:
        """
        Prompt the GPT model using Chain-of-thoughts with Langchain

        :return AIMessage
        """
        # Chat prompt
        prompt = self.gpt_prompt.create_prompt(request_sentence)

        # Chain of thoughts
        chain_of_thoughts = prompt | self.model | self.output_parser

        # Generate resposne
        llm_output = chain_of_thoughts.invoke({"input": request_sentence})
        
        # postprocess the response
        if llm_output is not []:
            predicted_class = llm_output['args']['pred']
        else:
            predicted_class = None

        return {"review": request_sentence, "predicted_class": predicted_class}

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def predict_with_tenacity(self, chat_prompt: List) -> AIMessage:
        """
        Prompt the GPT model using Chain-of-thoughts with Langchain tenacitly.

        :param chat_prompt List
            List of chat_prompt to the LLM

        :return AIMessage
        """

        # generate resposne
        self.predict(chat_prompt)
