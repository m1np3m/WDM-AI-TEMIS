
import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()

with open(here("configs/project_config.yml")) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)


class LoadProjectConfig:
    def __init__(self) -> None:

        # Load langsmith config
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = app_config["langsmith"]["tracing"]
        os.environ["LANGCHAIN_PROJECT"] = app_config["langsmith"]["project_name"]

        # Load memory config
        self.memory_dir = here(app_config["memory"]["directory"])
        self.embedding_model = app_config["embedding"]["model"]
        self.llm_model = app_config["llm"]["model"]


        self.mongo_uri = os.getenv("MONGO_URI")
        self.mongo_db = os.getenv("MONGO_DB")
        self.mongo_collection = os.getenv("MONGO_COLLECTION")
