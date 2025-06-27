from fastembed.embedding import TextEmbedding
import requests
import json
from dotenv import load_dotenv, find_dotenv
import os
import json
import re
import pandas as pd
import qdrant_client
from qdrant_client.models import VectorParams, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
import tqdm.notebook as tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from langchain.prompts import PromptTemplate
import random
import base64
from pydantic import BaseModel