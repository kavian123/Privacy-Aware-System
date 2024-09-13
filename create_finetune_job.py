from openai import OpenAI
from dotenv import load_dotenv
import os
import openai

load_dotenv()
client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )


#to finetune a model
file_id = "file-id"

x = client.fine_tuning.jobs.create(
  training_file=file_id,
  model="gpt-4o-mini-2024-07-18"# gpt-3.5-turbo-0125 #gpt-4o-2024-08-06 #gpt-4o-mini-2024-07-18
)

print(x)
'''
y = client.fine_tuning.jobs.list(limit=10)


print(y)
'''
