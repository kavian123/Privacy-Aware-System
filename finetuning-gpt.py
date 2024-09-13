import os
import openai
import random
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import json
import pandas as pd


load_dotenv()
client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

N_RETRIES = 3


prompt = ("A model acting as a healthcare assistive robot that takes in a user prompt (either requesting data access,"
          " update health guidelines, or authorise new user), and thoughtfully respond"
          " always in JSON format with at least one entity (key) only from each entity_classes, entity_classes=['Data_Type', 'Request_Type', 'User', 'Details', 'Inclusions', 'Exclusions'] "
          "'User' entity can be 'Null'."
          "where the entities for each classes are: "
          "Data_Type={'System_Information': 'Information related to the system', 'Heart_Rate': 'Information related to the heart rate of the patient', 'Steps': 'Information related to the steps activity  or any physical activity of the patient'} "
          "Request_Type={'Data_Access': 'User requesting access to a Data_Type', 'Guideline_Update': 'Request to modify Threshold_Heart_Rate or Steps_Goals', 'Authorise': 'User that needs to be authorised to access the data, this can be 'nurse', 'carer', 'doctor', 'engineer', 'family' or the user_ID'} "
          "Details={'Average': 'Requesting mean value of the data for a specified period', 'Sum': 'Total count over a specified period', 'Max': 'Requesting maximum value of the data for a specified period', 'Min': 'Requesting minimum value of the data for a specified period', 'Threshold_Heart_Rate': 'Heart rate threshold set and modified through Guideline_Update', 'Steps_Goals': 'Step count goals set and modified through Guideline_Update', 'Consistency': 'requesting for the consistency of data collection over a specified period'} "
          "User={'User': 'The user which the information is requested for (not the one who is requesting the data), this can be either 'doctor', 'nurse', 'carer', 'engineer', 'family', or 'null' if not specified for which user.'}"
          "Inclusions={'Date_From': 'Start date of the requested data', 'Date_Till': 'End date of the requested data', 'Time_From': 'Start time of the requested data', 'Time_Till': 'End time of the requested data', 'NULL': 'If no inclusions or date and time specific data is requested'} "
          "Exclusions={'Date_From': 'Start date of the requested data that should be excluded', 'Date_Till': 'End date of the requested data that should be excluded', 'Time_From': 'Start time of the requested data that should be excluded', 'Time_Till': 'End time of the requested data that should be excluded', 'NULL': 'If no exclusion or exception is requested'} .")
temperature = .3
number_of_examples = 1

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 8:
            prev_examples = random.sample(prev_examples, 8)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=1000,
    )

    return response.choices[0].message.content


# Generate examples
prev_examples = []
for i in range(number_of_examples):
    print(f'Generating example {i}')
    #print(prompt)
    example = generate_example(prompt, prev_examples, temperature)
    prev_examples.append(example)

print(prev_examples)



def generate_system_message(prompt):

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
          {
            "role": "system",
            "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
          },
          {
              "role": "user",
              "content": prompt.strip(),
          }
        ],
        temperature=temperature,
        max_tokens=500,
    )

    return response.choices[0].message.content

system_message = generate_system_message(prompt)

print(f'The system message is: `{system_message}`. Feel free to re-run this cell if you want a better result.')



# Initialize lists to store prompts and responses
prompts = []
responses = []

# Parse out prompts and responses from examples
for example in prev_examples:
  try:
    split_example = example.split('-----------')
    prompts.append(split_example[1].strip())
    print(responses)
    responses.append(split_example[3].strip())
  except:
    pass

# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

# Remove duplicates
df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples.')

# Initialize list to store training examples
training_examples = []

# Create training examples in the format required for GPT fine-tuning
for index, row in df.iterrows():
    training_example = {
        "messages": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": row['response']}
        ]
    }
    training_examples.append(training_example)

# Save training examples to a .jsonl file
with open('testing_dataset2.jsonl', 'w') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')





