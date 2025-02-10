from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import numpy as np
import os

def create_noisy_background(size, base_color):
    # Create base image
    img = Image.new('RGB', size, base_color)
    pixels = np.array(img)
    
    # Add noise
    noise = np.random.randint(-20, 20, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(pixels)

def generate_bonus_captcha_dataset(num_samples=10000, output_dir='bonus_captcha_dataset'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors (RGB)
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0)
    }
    
    # List of fonts to use
    potential_fonts = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/Times.ttc',
        '/System/Library/Fonts/AmericanTypewriter.ttc',
        '/System/Library/Fonts/Arial.ttf',
        '/System/Library/Fonts/Georgia.ttf'
    ]
    
    available_fonts = [f for f in potential_fonts if os.path.exists(f)]
    if not available_fonts:
        raise Exception("No usable fonts found!")
    
    for i in range(num_samples):
        # Generate random text
        word_length = random.randint(5, 10)
        text = ''.join([
            random.choice(string.ascii_letters).upper() 
            if random.random() > 0.5 
            else random.choice(string.ascii_letters).lower() 
            for _ in range(word_length)
        ])
        
        # Select background color and create noisy background
        color_name = random.choice(['red', 'green'])
        img = create_noisy_background((400, 200), colors[color_name])
        draw = ImageDraw.Draw(img)
        
        # Select random font and size
        font_size = random.randint(40, 60)
        font = ImageFont.truetype(random.choice(available_fonts), font_size)
        
        # Determine display text based on background color
        display_text = text[::-1] if color_name == 'red' else text
        
        # Center text
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (400 - text_width) // 2
        y = (200 - text_height) // 2
        
        # Draw text with slight shadow for better visibility
        draw.text((x+2, y+2), display_text, font=font, fill='black')
        draw.text((x, y), display_text, font=font, fill='white')
        
        # Apply random blur
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        
        # Save with original text in filename
        filename = f'{output_dir}/captcha_{text}_{color_name}_{i}.png'
        img.save(filename)
        
        print(f'Generated {filename}')

if __name__ == '__main__':
    try:
        generate_bonus_captcha_dataset()
    except Exception as e:
        print(f"Error generating captchas: {str(e)}")
