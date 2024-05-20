import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables from .env file
load_dotenv()

# Get the API token from environment variable
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Get the prompt from the command line argument
prompt = "Send an email from andreas86@telia.se to myfriend@telia.se where you discuss the weather. In the body, describe the current weather in Stockholm as detailed as possible.\n\nRespond with json that adheres to the following jsonschema:\n\n{jsonschema}"

input = {
    "prompt": prompt,
    "grammar": "",
    "jsonschema": """
    {
        "type": "object",
        "properties": {
            "Id": {
                "type": "integer"
            },
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
            "gender": {
                "type": "string"
            },
            "diagnosis": {
                "type": "string"
            }
        }
    }
    """
}

llm = Replicate(
    model="andreasjansson/llama-2-13b-chat-gguf:60ec5dda9ff9ee0b6f786c9d1157842e6ab3cc931139ad98fe99e08a35c5d4d4",
    model_kwargs={
        "temperature": 0.75,
        "max_length": 500,
        "top_p": 1,
        "jsonschema": input["jsonschema"]
    },
)
llm.invoke(prompt)

llm = Replicate(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model="andreasjansson/llama-2-13b-chat-gguf:60ec5dda9ff9ee0b6f786c9d1157842e6ab3cc931139ad98fe99e08a35c5d4d4",
    model_kwargs={
        "temperature": 0.75,
        "max_length": 500,
        "top_p": 1,
        "jsonschema": input["jsonschema"]
    },
)
_ = llm.invoke(prompt)

