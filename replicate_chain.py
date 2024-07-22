import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_community.llms import Replicate, Ollama 
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
    "patient_name": {
      "type": "string"
    },
    "patient_age": {
      "type": "integer"
    },
    "patient_gender": {
      "type": "string"
    },
    "medical_diagnosis": {
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
    "medical_treatment": {
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
    "health_vital": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "status": {
            "type": "string"
          },
          "value": {
            "type": "string"
          },
          "units": {
            "type": "string"
          }
        },
        "required": ["status"]
      }
    }
  },
  "required": ["name", "gender", "medical_treatment", "health_vital"]
}"""

system_prompt = f"""You are an AI that summarizes medical conversations into a structured JSON format like this{json_schema}. 
Given the medical transcript below, provide a medical summary by extracting key-value pairs. Only use the information explicitly mentioned 
in the transcript, and you must not infer or assume any details that are not directly stated. Strictly follow what the json schema required, 
and print the JSON schema only. If the transcript has no medical information, you must still proceed to print out a JSON schema. 
You must ensure that the 'medical_diagnosis', 'medical_treatment', and 'health_vital' fields contain valid medical terms recognized in medical practice."""

input_transcript = """
Speaker 1: Good morning. Could you please update me on the status of our project, Mr. Anderson? 
Speaker 2: Good morning. Mr. Anderson’s project progress has stabilized and is on track, even though he is already 70 years old. His report this morning showed an 8% increase in efficiency. 
Speaker 1: That’s good to hear. Has he completed his tasks for the morning? 
Speaker 2: Yes. He finished his treatment at 7 AM, as scheduled. 
Speaker 1: Excellent. Please continue to monitor his progress and let me know if there are any significant developments. 
Speaker 2: Absolutely. I’ll keep you informed of any updates.
"""
prompt = f"""
System: {system_prompt}
User: {input_transcript}
Assistant:
"""
# llm = Replicate(
#     streaming=True,
#     callbacks=[StreamingStdOutCallbackHandler()],
#     model="meta/meta-llama-3-70b-instruct",
#     model_kwargs= {
#         "top_k": 0,
#         "top_p": 0.9,
#         "max_tokens": 512,
#         "min_tokens": 0,
#         "temperature": 0.6,
#         "length_penalty": 1,
#         "stop_sequences": "<|end_of_text|>,<|eot_id|>",
#         "presence_penalty": 1.15,
#         "log_performance_metrics": False}, 
#         )
llm = Ollama(model="llama3:70b", temperature=0)
_ = llm.invoke(prompt)

