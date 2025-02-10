from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import numpy as np
import os

def create_textured_background(size):
    """Create a noisy, textured background"""
    # Create base image with random light color
    base_color = (
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255)
    )
    img = Image.new('RGB', size, base_color)
    pixels = np.array(img)
    
    # Add noise
    noise = np.random.randint(-30, 30, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(pixels)

def generate_hard_captcha_dataset(num_samples=40000, output_dir='hard_captcha_dataset'):
    os.makedirs(output_dir, exist_ok=True)
    
    # List of fonts to try
    potential_fonts = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/Times.ttc',
        '/System/Library/Fonts/AmericanTypewriter.ttc',
        '/System/Library/Fonts/Georgia.ttf',
        '/System/Library/Fonts/Arial.ttf'
    ]
    
    available_fonts = [f for f in potential_fonts if os.path.exists(f)]
    if not available_fonts:
        raise Exception("No usable fonts found!")

    for i in range(num_samples):
        img = create_textured_background((400, 200))
        draw = ImageDraw.Draw(img)
        
        # Generate text
        word_length = random.randint(4, 8)
        text = ''.join([
            random.choice(string.ascii_letters).upper() 
            if random.random() > 0.5 
            else random.choice(string.ascii_letters).lower() 
            for _ in range(word_length)
        ])
        
        # Fixed dimensions and padding
        char_padding = 5
        edge_padding = 40
        available_width = 400 - (2 * edge_padding)
        
        # Calculate font size based on word length
        min_font_size = 25
        max_font_size = min(45, available_width // (word_length + 2))  # +2 for padding
        if max_font_size < min_font_size:
            max_font_size = min_font_size
        
        total_width = 0
        char_info = []
        
        # First pass: calculate sizes
        for char in text:
            # Use safe font size range
            font_size = random.randint(min_font_size, max_font_size)
            font = ImageFont.truetype(random.choice(available_fonts), font_size)
            
            # Get exact character dimensions
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            
            char_info.append({
                'char': char,
                'font': font,
                'width': char_width,
                'height': char_height,
                'bbox': bbox,
                'spacing': char_padding
            })
            total_width += char_width + char_padding
        
        # Calculate positions
        max_height = max(info['height'] for info in char_info)
        y_position = (200 - max_height) // 2
        start_x = (400 - total_width) // 2
        current_x = start_x
        
        # Second pass: render characters
        for info in char_info:
            # Create precise-sized image for character
            char_img = Image.new('RGBA', 
                               (info['width'], info['height']), 
                               (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            
            # Center character precisely
            char_x = -info['bbox'][0]  # Adjust for left bearing
            char_y = -info['bbox'][1]  # Adjust for top bearing
            
            # Darker text color for better visibility
            text_color = (
                random.randint(0, 60),
                random.randint(0, 60),
                random.randint(0, 60)
            )
            
            # Draw character
            char_draw.text((char_x, char_y), 
                         info['char'], 
                         font=info['font'], 
                         fill=text_color)
            
            # Paste character with precise positioning
            paste_x = current_x
            paste_y = y_position + ((max_height - info['height']) // 2)
            img.paste(char_img, (paste_x, paste_y), char_img)
            
            # Update position for next character
            current_x += info['width'] + info['spacing']
        
        # Apply effects with reduced intensity
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
        
        if random.random() < 0.3:
            img = img.filter(ImageFilter.EDGE_ENHANCE)
        
        filename = f'{output_dir}/captcha_{text}_{i}.png'
        img.save(filename)
        print(f'Generated {filename}')

if __name__ == '__main__':
    try:
        generate_hard_captcha_dataset()
    except Exception as e:
        print(f"Error generating captchas: {str(e)}")
