import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import json
import re
from PIL import Image as PILImage
from io import BytesIO
import requests

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configuration
STORY_PROMPT = """
Create an engaging story that is rich in visual descriptions. The story should be structured 
with clear scene transitions and vivid imagery. Include detailed descriptions of:
- Characters and their appearances
- Settings and environments
- Actions and movements
- Atmospheric elements (lighting, weather, mood)

The story should be between 500-1000 words.
"""

# Base template for image generation
IMAGE_PROMPT_TEMPLATE = """
{scene_description}, digital art style, dreamy atmosphere, vibrant colors, 
highly detailed, 4k, masterpiece quality, consistent lighting, cinematic composition, 
professional photography, artstation trending
"""

def generate_story():
    """Generate a story using OpenAI's GPT model"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a creative writer who excels at creating visually descriptive stories."},
                {"role": "user", "content": STORY_PROMPT}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating story: {e}")
        return None

def split_story_into_segments(story):
    """Split the story into segments for image generation"""
    # Split by paragraphs
    segments = [s.strip() for s in story.split('\n\n') if s.strip()]
    
    # Process each segment to extract key visual elements
    processed_segments = []
    for segment in segments:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract the key visual elements from this text segment. Focus on describing the scene, characters, and atmosphere in a way that could be turned into an image."},
                    {"role": "user", "content": segment}
                ],
                temperature=0.3
            )
            visual_summary = response.choices[0].message.content
            processed_segments.append({
                'text': segment,
                'visual_summary': visual_summary
            })
        except Exception as e:
            print(f"Error processing segment: {e}")
            processed_segments.append({
                'text': segment,
                'visual_summary': segment
            })
    
    return processed_segments

def generate_image(visual_summary):
    """Generate an image for a story segment using DALL-E"""
    try:
        # Combine the visual summary with our base template
        prompt = IMAGE_PROMPT_TEMPLATE.format(scene_description=visual_summary)
        
        response = client.images.generate(
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        
        # Download the image
        image_url = response.data[0].url
        response = requests.get(image_url)
        img = PILImage.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def save_image(img, filename):
    """Save the generated image to a file"""
    try:
        img.save(filename)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def main():
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Generate the story
    print("Generating story...")
    story = generate_story()
    if not story:
        return
    
    # Save the story
    with open("output/story.txt", "w", encoding="utf-8") as f:
        f.write(story)
    
    print("\nGenerated Story:")
    print(story)
    
    # Split the story into segments
    print("\nSplitting story into segments...")
    segments = split_story_into_segments(story)
    
    # Generate and save images for each segment
    print("\nGenerating images for each segment...")
    for i, segment in enumerate(segments, 1):
        print(f"\nSegment {i}:")
        print("Text:")
        print(segment['text'])
        print("\nGenerating image...")
        img = generate_image(segment['visual_summary'])
        if img:
            save_image(img, f"output/segment_{i}.png")
        time.sleep(2)  # Rate limiting

if __name__ == "__main__":
    main() 