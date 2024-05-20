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
json_schema="""{
  "type": "object",
  "properties": {
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
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          }
        },
        "required": ["name"]
      }
    },
    "treatment": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "prescription": {
            "type": "string"
          }
        },
        "required": ["name", "prescription"]
      }
    },
    "vital": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "value": {
            "type": "string"
          },
          "units": {
            "type": "string"
          }
        },
        "required": ["name", "value", "units"]
      }
    }
  },
  "required": ["name", "gender", "treatment", "vital"]
}"""

system_prompt = f"""You are an AI that summarizes medical conversations into a structured JSON format like this{json_schema}. 
Given the medical transcript below, provide a summary by extracting key-value pairs. Only use the information explicitly mentioned 
in the transcript, and you must not infer or assume any details that are not directly stated, and strictly follow what the json schema required, 
and print the json schema only."""

input_transcript = """
Speaker 1: Good morning, Nurse. Could you please update me on the status of our patient, Mr. Anderson?
Speaker 2: Good morning, Doctor. Mr. Anderson's blood pressure has stabilized at 100/70. His blood glucose level this morning was 8 mmol/L.
Speaker 1: That's good to hear. Has he taken his medication for the morning?
Speaker 2: Yes, Doctor. He took his Metformin at 7 AM, as prescribed.
Speaker 1: Excellent. Please continue to monitor his vitals and let me know if there are any significant changes.
Speaker 2: Absolutely, Doctor. I'll keep you updated.
"""
prompt = f"""
System: {system_prompt}
User: {input_transcript}
Assistant:
"""
llm = Replicate(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model="meta/meta-llama-3-70b-instruct",
    model_kwargs= {
        "top_k": 0,
        "top_p": 0.9,
        "max_tokens": 512,
        "min_tokens": 0,
        "temperature": 0.6,
        "length_penalty": 1,
        "stop_sequences": "<|end_of_text|>,<|eot_id|>",
        "presence_penalty": 1.15,
        "log_performance_metrics": False}, 
        )

_ = llm.invoke(prompt)

