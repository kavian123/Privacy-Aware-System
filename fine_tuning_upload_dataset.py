from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

x = client.files.create(
  file=open("training_dataset.jsonl", "rb"),
  purpose="fine-tune"
)

print(x)
