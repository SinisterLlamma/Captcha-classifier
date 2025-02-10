from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import numpy as np
import os

def create_textured_background(size):
    """Create a noisy, textured background"""
    base_color = (
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255)
    )
    img = Image.new('RGB', size, base_color)
    pixels = np.array(img)
    noise = np.random.randint(-30, 30, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)

def get_text_variation(text):
    """Create a variation of text with random capitalization"""
    return ''.join(
        char.upper() if random.random() > 0.5 else char.lower()
        for char in text
    )

def generate_hard_variations_dataset(num_words=100, variations_per_word=200, output_dir='hard_variations_dataset'):
    os.makedirs(output_dir, exist_ok=True)
    
    # List of fonts to try (using the expanded font list)
    potential_fonts = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/Times.ttc',
        '/System/Library/Fonts/AmericanTypewriter.ttc',
        '/System/Library/Fonts/Georgia.ttf',
        '/System/Library/Fonts/Arial.ttf',
        '/System/Library/Fonts/Verdana.ttf',
        '/System/Library/Fonts/Tahoma.ttf',
        '/System/Library/Fonts/Impact.ttf',
        '/System/Library/Fonts/Comic Sans MS.ttf',
        '/System/Library/Fonts/Courier New.ttf',
        '/System/Library/Fonts/Arial Black.ttf',
        '/System/Library/Fonts/Trebuchet MS.ttf',
        '/System/Library/Fonts/Palatino.ttc',
        '/System/Library/Fonts/Monaco.ttf',
        '/System/Library/Fonts/Gill Sans.ttc',
        '/System/Library/Fonts/Futura.ttc',
        '/System/Library/Fonts/Optima.ttc',
        '/Library/Fonts/Arial Bold.ttf',
        '/Library/Fonts/Times New Roman Bold.ttf',
        '/Library/Fonts/Calibri.ttf',
        '/Library/Fonts/Cambria.ttf',
    ]
    
    available_fonts = [f for f in potential_fonts if os.path.exists(f)]
    if not available_fonts:
        raise Exception("No usable fonts found!")

    # Generate word list first
    word_list = []
    for _ in range(num_words):
        word_length = random.randint(5, 10)
        text = ''.join([
            random.choice(string.ascii_letters).upper() 
            if random.random() > 0.5 
            else random.choice(string.ascii_letters).lower() 
            for _ in range(word_length)
        ])
        word_list.append(text)

    # Generate variations for each word
    for word_idx, base_text in enumerate(word_list):
        # Create label-specific directory
        label_dir = os.path.join(output_dir, base_text)
        os.makedirs(label_dir, exist_ok=True)

        for var_idx in range(variations_per_word):
            # Create variation of the text with different capitalization
            text = get_text_variation(base_text)
            
            img = create_textured_background((400, 200))
            draw = ImageDraw.Draw(img)
            
            # Fixed dimensions and padding
            char_padding = random.randint(3, 7)  # Vary character spacing
            edge_padding = random.randint(35, 45)  # Vary edge padding
            available_width = 400 - (2 * edge_padding)
            
            # Calculate font size based on word length
            min_font_size = 25
            max_font_size = min(45, available_width // (len(text) + 2))
            if max_font_size < min_font_size:
                max_font_size = min_font_size
            
            total_width = 0
            char_info = []
            
            # First pass: calculate sizes with different fonts per character
            for char in text:
                font_size = random.randint(min_font_size, max_font_size)
                # Select a different font for each character
                font = ImageFont.truetype(random.choice(available_fonts), font_size)
                
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
            
            # Calculate positions and render
            max_height = max(info['height'] for info in char_info)
            y_position = (200 - max_height) // 2
            start_x = (400 - total_width) // 2
            current_x = start_x
            
            # Second pass: render characters
            for info in char_info:
                char_img = Image.new('RGBA', (info['width'], info['height']), (0, 0, 0, 0))
                char_draw = ImageDraw.Draw(char_img)
                
                char_x = -info['bbox'][0]
                char_y = -info['bbox'][1]
                
                text_color = (
                    random.randint(0, 60),
                    random.randint(0, 60),
                    random.randint(0, 60)
                )
                
                char_draw.text((char_x, char_y), 
                             info['char'], 
                             font=info['font'], 
                             fill=text_color)
                
                paste_x = current_x
                paste_y = y_position + ((max_height - info['height']) // 2)
                img.paste(char_img, (paste_x, paste_y), char_img)
                
                current_x += info['width'] + info['spacing']
            
            # Apply effects
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
            
            if random.random() < 0.3:
                img = img.filter(ImageFilter.EDGE_ENHANCE)
            
            filename = os.path.join(label_dir, f'variation_{var_idx}.png')
            img.save(filename)
            print(f'Generated {filename} with text: {text}')

        print(f'Completed variations for base text: {base_text} ({word_idx + 1}/{len(word_list)})')

if __name__ == '__main__':
    try:
        generate_hard_variations_dataset()
    except Exception as e:
        print(f"Error generating captchas: {str(e)}")
