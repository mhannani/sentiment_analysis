from typing import List, Union
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate


class GPTPrompt:
    """
    Prompt for GPT model.
    """

    def __init__(self, config: object, fuzzy_matches: List = []) -> None:
        """
        Prompt class constructor

        :param fuzzy_matches List
            List of fuzzy matches

        :return None
        """

        # configuration object
        self.config = config

        # fuzzy_match
        self.fuzzy_matches: List = fuzzy_matches


    def create_prompt(self, request_sentence: str = None) -> Union[List, ChatPromptTemplate]:
        """
        Create a prompt given the sentence for GPT model.

        :param request_sentence str
            Request sentence

        :return str
        """

        # chat prompt template
        chat_messages = []

        # user request message template
        user_message_template = "{input}"

        # assisstant template
        template = self.config["prompting"]["system_message"]

        # create system message
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            template)

        # user message
        user_message_prompt = HumanMessagePromptTemplate.from_template(
            user_message_template)

        # format message with kwargs if provided
        if request_sentence is not None:
            user_message_prompt.format_messages(input=request_sentence)

        # add user request for translation
        chat_messages.append(user_message_prompt)

        # combine messages
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, *chat_messages])

        # format chat prompt
        # if request_sentence is not None:
        #     chat_prompt = chat_prompt.format_messages(input=request_sentence)

        return chat_prompt


if __name__ == "__main__":
    from pathlib import Path
    import os
    import sys

    # Get the path to the directory containing this script (src/prompts)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    sys.path.append(os.path.abspath(os.path.join(script_dir, "../..")))

    from src.utils.parsers import parse_toml

    # Change the working directory to the root directory of the project
    os.chdir("../..")

    # Load the configuration from the TOML file
    config = parse_toml(Path("./configs/config.toml"))

    # Instantiate the Prompt class with the loaded configuration
    prompt = GPTPrompt(config)

    # create prommpt with request sentence
    chat_prompt = prompt.create_prompt("Hello") # For List of messages # or chat_prompt = prompt.create_prompt() # to be as <class 'langchain_core.prompts.chat.ChatPromptTemplate'>

    print("chat prompt: ", type(chat_prompt))





