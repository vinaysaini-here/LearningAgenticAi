from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

api_key = os.getenv("Gemini_API_Key")
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[       
        {
            "role": "user",
            "content": "Hey I am Vinay Saini , Tell me a joke"
        }
    ]
)

print(response.choices[0].message)