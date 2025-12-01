from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv(override=True)
llm = AzureChatOpenAI(
    azure_endpoint="https://api.truhn.ai",
    api_key=os.getenv("OPENAI_API_KEY"),
    deployment_name="gpt-5",  # MUST MATCH AZURE DEPLOYMENT
    api_version="2024-10-21",
)

emb = AzureOpenAIEmbeddings(
    azure_endpoint="https://api.truhn.ai",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-10-21",
)