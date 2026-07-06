from dotenv import load_dotenv
import os
from google import genai

# Load API key from .env
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("=" * 50)
print("🤖 Gemini Chat Started")
print("Type 'exit' to quit")
print("=" * 50)

while True:

    # Take input from user
    question = input("\nYou: ")

    # Exit condition
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Send question to Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question
    )

    # Print answer
    print("\nGemini:")
    print(response.text)