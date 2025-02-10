from PIL import Image, ImageDraw, ImageFont
import random
import string
import os

def generate_easy_captcha_dataset(num_samples=10000, output_dir='easy_captcha_dataset'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a consistent, fixed font
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    font_size = 40
    font = ImageFont.truetype(font_path, font_size)
    
    for i in range(num_samples):
        # Generate random length between 5 and 10
        word_length = random.randint(5, 10)
        
        # Generate text with random length, first letter uppercase
        first_char = random.choice(string.ascii_uppercase)
        rest_chars = ''.join(random.choices(string.ascii_lowercase, k=word_length-1))
        text = first_char + rest_chars
        
        # Create a white background image
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Calculate text position to center it
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (400 - text_width) // 2
        y = (200 - text_height) // 2
        
        # Draw text in black
        draw.text((x, y), text, font=font, fill='black')
        
        # Save image with label in filename
        filename = f'{output_dir}/captcha_{text}_{i}.png'
        img.save(filename)
        
        print(f'Generated {filename}')

# Generate the dataset
generate_easy_captcha_dataset()
