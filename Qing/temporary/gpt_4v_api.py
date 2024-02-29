import base64
import requests

# OpenAI API Key
# api_key = "YOUR_OPENAI_API_KEY"
api_key="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"
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "../data/val2014/COCO_val2014_000000353096.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Decide whether each sentence in the following passage describes the picture correctly. Correct sentences are represented by True, and incorrect sentences are represented by False. The content of this passage is: The image features a white computer monitor sitting on a desk, with a keyboard and mouse placed in front of it. The monitor is turned on, displaying a blue screen with a lightning bolt on it. The keyboard and mouse are positioned to the right of the monitor, with the keyboard occupying a larger portion of the desk. The desk appears to be a wooden surface, providing a suitable workspace for the computer setup."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json()['choices'][0]['message']['content'])
