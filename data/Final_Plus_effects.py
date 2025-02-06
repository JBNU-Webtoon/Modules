import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Create output folder if not exists
def create_output_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

# Load effects from text file
def load_effects_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Effects file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        effects = [line.strip() for line in file if line.strip()]
    if not effects:
        raise ValueError("Effects file is empty.")
    return effects

# Generate random RGB color
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Calculate bounding box size after rotation
def calculate_rotated_bbox_size(width, height, angle):
    radians = np.radians(angle)
    new_width = abs(width * np.cos(radians)) + abs(height * np.sin(radians))
    new_height = abs(width * np.sin(radians)) + abs(height * np.cos(radians))
    return int(new_width), int(new_height)

# Ensure the text stays within image boundaries
def calculate_safe_position(image_width, image_height, rotated_width, rotated_height, padding):
    max_x = image_width - rotated_width - padding
    max_y = image_height - rotated_height - padding
    return (
        random.randint(padding, max(padding, max_x)),
        random.randint(padding, max(padding, max_y))
    )

# Select random font from folder
def random_font_from_folder(font_folder):
    font_files = [file for file in os.listdir(font_folder) if file.endswith('.ttf')]
    if not font_files:
        raise FileNotFoundError("No font files found in the folder.")
    return os.path.join(font_folder, random.choice(font_files))

# Add random spaces to a string
def add_random_spaces(text):
    return "".join(char + " " * random.randint(0, 2) for char in text).strip()

def save_cropped_effect(output_path, effect_text, bbox, effects_folder, labels_file):
    """
    Crop the effect text area from the image and save it with its label.
    
    ARGS:
        image_path: Path to the original image.
        effect_text: The text used for the effect.
        bbox: The bounding box (x, y, width, height) of the effect.
        effects_folder: Folder to save cropped effect images.
        labels_file: File to save the labels for cropped images.
    """
    # Load the image
    image = cv2.imread(output_path)
    x, y, width, height = bbox
    x, y, width, height = int(x), int(y), int(width), int(height)

    # Crop the effect area
    cropped_effect = image[y:y+height, x:x+width]
    effect_image_name = f"{os.path.splitext(os.path.basename(output_path))[0]}.png"

    # Save the cropped effect
    cropped_effect_path = os.path.join(effects_folder, effect_image_name)
    cv2.imwrite(cropped_effect_path, cropped_effect)

    # Save label information
    with open(labels_file, 'a', encoding='utf-8') as f:
        f.write(f"{cropped_effect_path} {effect_text}\n")


# Add random text to an image and save cropped effect
def add_random_text_with_crop(image_path, output_path, font_folder, effects_file, bbox_file, effects_folder, labels_file):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Load effects and choose random text
    effects = load_effects_from_file(effects_file)
    text = add_random_spaces(random.choice(effects))

    # Set font and initial size
    font_path = random_font_from_folder(font_folder)
    font_size = int(pil_image.width * 0.3)
    font = ImageFont.truetype(font_path, font_size)

    # Calculate text dimensions and adjust size
    image_width, image_height = pil_image.size
    padding = int(min(image_width, image_height) * 0.1)

    while True:
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
        angle = random.uniform(-10, 10)
        rotated_width, rotated_height = calculate_rotated_bbox_size(text_width, text_height, angle)
        x, y = calculate_safe_position(image_width, image_height, rotated_width, rotated_height, padding)

        if x + rotated_width + padding <= image_width and y + rotated_height + padding <= image_height:
            break
        font_size = max(font_size - 10, int(pil_image.width * 0.05))
        font = ImageFont.truetype(font_path, font_size)

    # Draw text with outline
    text_image = Image.new("RGBA", (rotated_width, rotated_height), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_image)
    outline_color = (0, 0, 0) if random_color() == (255, 255, 255) else (255, 255, 255)

    for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
        text_draw.text((dx, dy), text, font=font, fill=outline_color)
    text_draw.text((0, 0), text, font=font, fill=random_color())

    # Rotate and paste text
    rotated_text_image = text_image.rotate(angle, resample=Image.BICUBIC, expand=True)
    pil_image.paste(rotated_text_image, (x, y), rotated_text_image)

    # Save result image
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_image)

    # Calculate bounding box
    bbox = (x, y, rotated_width, rotated_height)

    # Save bounding box data in YOLO format
    with open(bbox_file, 'a', encoding='utf-8') as bbox_file:
        bbox_file.write(f"{os.path.basename(image_path)} {bbox[0]} {bbox[1]} {x+bbox[2]} {y+bbox[3]} 0\n")

    # Save cropped effect and its label
    save_cropped_effect(output_path, text, bbox, effects_folder, labels_file)


# Generate novel images with effects and save bounding boxes and cropped effects
def make_novel_with_crops(image_folder, output_folder, font_folder, effects_file, bbox_file, effects_folder, labels_file):
    create_output_folder(output_folder)
    create_output_folder(effects_folder)
    image_files = [file for file in os.listdir(image_folder) if file.endswith(('.jpeg', '.png', '.jpg', 'JPEG'))]
    if not image_files:
        raise FileNotFoundError("No image files found in the folder.")

    # Clear bbox and labels files if they exist
    open(bbox_file, 'w').close()
    open(labels_file, 'w').close()

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        add_random_text_with_crop(image_path, output_path, font_folder, effects_file, bbox_file, effects_folder, labels_file)

    print(f"All bounding box data saved to {bbox_file}")
    print(f"All cropped effects saved to {effects_folder}")
    print(f"Labels saved to {labels_file}")


# Example Usage
make_novel_with_crops(
    image_folder="./Images",
    output_folder="./outputs",
    font_folder="./Fonts",
    effects_file="./effects.txt",
    bbox_file="./bbox.txt",
    effects_folder="./effects",
    labels_file="./labels.txt"
)

