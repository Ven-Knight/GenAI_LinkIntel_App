import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print("✅ API key is working. Response received:")
    print(response.choices[0].message["content"])
except Exception as e:
    print("❌ API key test failed:")
    print(e)