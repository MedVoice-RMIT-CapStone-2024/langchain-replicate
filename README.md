# LangChain x Replicate API
This script uses the Replicate class from the langchain_community.llms module to generate responses to a given prompt. The responses adhere to a specified JSON schema.
## Build instructions
### To run the script
- Initialize the virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
- Install the necessary dependencies
```bash
pip install -r requirements
```
- Run the script
```bash
python replicate_chain.py
```

### In Jupyter Notebook
- After running `REPLICATE_API_TOKEN = getpass()` Put in the API token to the top bar in VSCode
