from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_openai(SYSTEM_PROMPT, USER_PROMPT, model="gpt-4o-mini", temperature=0):

    try:
        # Print the prompt being sent
        # print(f"Prompt Sent to Model:\n{USER_PROMPT}\n")

        response = oai_client.chat.completions.create(
            # model="meta-llama/Llama-3-8b-chat-hf",  # Latest model version
            # model="mistralai/Mistral-7B-Instruct-v0.3",
            # model="google/gemma-2-9b-it",
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": USER_PROMPT
                }
            ],
            # max_tokens=None,
            temperature=temperature,  # For deterministic output
        )
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    return response.choices[0].message.content.strip()


# 1792x1024, 1024x1792
def generate_image(prompt):
    response = oai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        response_format="url",
    )
    return response.data[0].url


def generate_audio(prompt, name):
    response = oai_client.audio.speech.create(
        model="tts-1",
        input=prompt,
        voice="alloy",
    )
    response.stream_to_file(f"outputs/{name}.mp3")

    return f"outputs/{name}.mp3"
