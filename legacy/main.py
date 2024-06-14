from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap
import wikipedia
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re


def extract_hindi_words(line):
    hindi_words = re.findall(r'\b[\u0900-\u097F]+\b', line)
    return hindi_words

def extract_assamese_words(line):
    assamese_words = re.findall(r'\b[\u0980-\u09FF]+\b', line)
    return assamese_words

def extract_english_words(line):
    english_words = re.findall(r'\b[a-zA-Z]+\b', line)
    return english_words

def thin_pg_blur_img(image, avg_height, std_dev):
    """
    Blur the input image with a specified blur level.

    Args:
    - image: PIL image.
    - blur_level: Level of blur (integer). Higher values result in stronger blur. Default is 1.

    Returns:
    - Blurred image (PIL image).
    """
    
    image_np = np.array(image)
    sigma = std_dev
    
    blurred_image_np = cv2.GaussianBlur(image_np, (2 * (avg_height) + 1 , 4 * (avg_height) + 1 ), sigma) 

    # # Define parameters for Gaussian blur
    image_size = (2 * (avg_height) + 1, 2 * (avg_height) + 1)  # Size of the image
    # Standard deviation of the Gaussian kernel

    # Generate a Gaussian kernel using cv2.GaussianBlur
    gaussian_kernel = cv2.getGaussianKernel(image_size[0], sigma)

    # Plot the Gaussian window
    plt.figure(figsize=(8, 4))
    plt.plot(gaussian_kernel)
    plt.title('Gaussian Window (cv2.GaussianBlur)')
    plt.xlabel('Pixel')
    plt.ylabel('Amplitude')
    plt.grid(True)
    # plt.show()
    
    # Convert back to PIL image
    blurred_image = Image.fromarray(blurred_image_np)
    print(avg_height)

    return blurred_image

def blur_image(image, blur_level):
    """
    Blur the input PIL image with a specified blur level.
    
    Args:
    - image: PIL.Image - The input image to be blurred.
    - blur_level: int - The level of blur to be applied. Larger values result in more blur.
    
    Returns:
    - PIL.Image - The blurred image.
    """
    # Apply Gaussian blur filter to the image
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_level))
    
    return blurred_image



def binarize_img(image, threshold=220):
    """
    Binarize the input image with a specified threshold.

    Args:
    - image: PIL image.
    - threshold: Threshold value (integer). Pixels below this value will be set to black (0), and pixels above
                 this value will be set to white (255). Default is 128.

    Returns:
    - Binarized image (PIL image).
    """
    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Convert the image to grayscale
    gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply thresholding
    _, binarized_image_np = cv2.threshold(gray_image_np, threshold, 255, cv2.THRESH_BINARY)

    # Convert back to PIL image
    binarized_image = Image.fromarray(binarized_image_np)

    return binarized_image



def get_random_page_content() -> str:
    """
    Get a random Wikipedia page content.
    """
    page_title = wikipedia.random(1)
    try:
        page_content = wikipedia.page(page_title).summary
    except (wikipedia.DisambiguationError, wikipedia.PageError):
        return get_random_page_content()
    return page_content

def create_strings_from_wikipedia(lang: str, minimum_length: int, count: int ) -> list:
    """
    Create a list of strings from random Wikipedia articles.
    """
    wikipedia.set_lang(lang)
    sentences = []

    while len(sentences) < count:
        page_content = get_random_page_content()
        processed_content = page_content.replace("\n", " ").split(". ")
        sentence_candidates = [
            s.strip() for s in processed_content if len(s.split()) > minimum_length
        ]
        sentences.extend(sentence_candidates)

    return sentences[:count]

def read_random_lines_from_file(filename, num_lines):
    """
    Read a specified number of random lines from a text file and return as a list.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    # Remove newline characters from each line
    lines = [line.strip() for line in lines]
    # Randomly select specified number of lines
    selected_lines = random.sample(lines, num_lines)
    return selected_lines


def generate_word_bbox(image_save_name:str, word_box_save_path: str, words_with_bboxes: list) -> None:
    """
    Generate and write word bounding boxes to text files inside the output folder.
    """

    output_file = os.path.join(word_box_save_path, f"{image_save_name}.txt")
    
    with open(output_file, 'w') as f:
        for word, bbox in words_with_bboxes:
            bbox_str = ','.join(str(coord) for coord in bbox)
            f.write(f"{bbox_str}    {word}\n")


def generate_line_bbox(image_save_name:str, line_box_save_path: str, lines_with_bboxes: list) -> None:
    """
    Generate and write line bounding boxes to the output file.
    """

    output_file = os.path.join(line_box_save_path, f"{image_save_name}.txt") 

    with open(output_file, 'w') as f:
        for line, line_bbox in lines_with_bboxes:
            line_bbox_str = ','.join(str(coord) for coord in line_bbox)
            f.write(f"{line_bbox_str}\n")


def generate_text_image(image_save_name: str, gen_folder_path: str, lang : str ,lines: list, background_image_path: str, line_box_save_path: str, word_box_save_path: str,
                        blur: int, color_code: tuple, font_path: str , font_size: int, word_space: int, line_space: int, margin: int, margin_bottom: int, margin_top: int) -> list:
    """
    Generate an image with text overlaid on a background image and save bounding boxes of words.
    """

    img_width, img_height = 600, 800
    max_word_height = 0
    max_text_width = img_width - margin  # Max width for text
    max_text_height = img_height - margin_top #Max height for text

    #Creating Images
    img = Image.open(background_image_path).resize((img_width, img_height)) # Main image
    img_white_bg = Image.new('RGB', (img_width, img_height), color='white')  # Image with White Background

    #Functions to draw text over image
    draw = ImageDraw.Draw(img)
    draw_white_bg = ImageDraw.Draw(img_white_bg)

    font = ImageFont.truetype(font_path if font_path else ImageFont.load_default(), font_size)

    words_with_bboxes = []
    single_line = []
    lines_with_bboxes = []

    cnt = 0

    x_start_per_word = margin
    y_start_per_word = margin_top

    for line in lines:
        if y_start_per_word+margin_bottom > max_text_height:
            break

        if lang == 'en':
            english_words = extract_english_words(line)
            filtered_lines = ' '.join(english_words)
        
        words = filtered_lines.split()

        line_width = 0 

        for word in words:
            
            cnt += 1
            word_width, word_height = draw.textsize(word, font=font)
            max_word_height = max(max_word_height, word_height)


            if x_start_per_word + word_width > max_text_width:
                final_line = ' '.join(char for char in single_line)
                lines_with_bboxes.append((final_line, (margin, y_start_per_word, x_start_per_word - word_space, y_start_per_word + max_word_height)))
                
                single_line = []
                line_width = 0
                x_start_per_word = margin
                y_start_per_word += (max_word_height + line_space)
                max_word_height = 0
                if y_start_per_word+margin_bottom > max_text_height:
                    break

            word_bbox = (x_start_per_word, y_start_per_word, x_start_per_word + word_width, y_start_per_word + word_height)  # Bounding Box Coordinates
            words_with_bboxes.append((word, word_bbox))
            single_line.append(word)
            
            draw.text((x_start_per_word, y_start_per_word), word, 0, font=font) # Draw text over main image
            draw_white_bg.text((x_start_per_word, y_start_per_word), word, 0, font=font) # Draw text over white background

            line_width = line_width + (word_width + word_space)
            x_start_per_word = x_start_per_word + (word_width + word_space)

    final_line = ' '.join(char for char in single_line)
    lines_with_bboxes.append((final_line, (margin, y_start_per_word, x_start_per_word - word_space, y_start_per_word + max_word_height)))

    img = blur_image(img, blur)
    save_image_with_unique_name(image_save_name, image_save_path, img)

    generate_line_bbox(image_save_name, line_box_save_path, lines_with_bboxes)
    generate_word_bbox(image_save_name, word_box_save_path, words_with_bboxes)


def save_image_with_unique_name(image_save_name, image_save_path, image):

    filename = os.path.join(image_save_path, f"{image_save_name}.png")
    image.save(filename)
    print(f"Image saved as: {filename}")


def choose_background_image():
    images_folder = 'background/'  # Specify the folder containing your background images
    background_images = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith(('.jpeg', '.jpg', '.png'))]
    return background_images[random.randint(0, len(background_images) - 1)]

def choose_font(language : str):
    fonts_folder = 'fonts/fonts/' + lang   
    fonts = [os.path.join(fonts_folder, f) for f in os.listdir(fonts_folder) if f.endswith(('.ttf','.otf'))]
    return fonts[random.randint(0, len(fonts) - 1)]

def choose_text_color():
    return (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), 255)
    # return (0, 0, 0, 255)

def choose_blur_level():   
    return random.uniform(0.5, 1)

def choose_language():
    languages = ["en"]
    return random.choice(languages)


def choose_text_size():
    return random.randint(5, 20)

def choose_word_space():
    return random.randint(3, 10)

def choose_line_space():
    return random.randint(6, 10)

def choose_lines_read():
    return random.randint(1, 200)

def choose_margin():
    return random.randint(6, 20)

def random_bottom_margin():
    return random.randint(6, 20)

def random_top_margin():
    return random.randint(6, 20)

def alter_image(background_image_path):
    """
    Alter the Background Intensity of Pixels
    """
    image = cv2.imread(background_image_path)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Could not open or find the image.")
        exit()


    # Define the value by which you want to increase the intensity
    increase_value = 50

    # Increase the intensity by adding the increase_value to each pixel intensity
    increased_image = np.clip(image.astype(np.int32) + increase_value, 0, 255).astype(np.uint8)

    # Display the original and increased intensity images
    # cv2.imsave('Original Image', image)
    # cv2.imwrite('altered_bg.png', increased_image)
    return increased_image

image_save_path = 'output/images'
line_box_save_path = 'output/line_bbox'
word_box_save_path = 'output/word_bbox'
Lines = "text/text/file/with/words.txt"


for i in range(100):
    blur = choose_blur_level()
    lang = choose_language()
    color_code = choose_text_color()
    font_path = choose_font(lang)
    background_image_path = choose_background_image()
    font_size=choose_text_size()
    word_space=choose_word_space()
    line_space=choose_line_space()
    number_of_lines_to_read = choose_lines_read()
    margin = choose_margin()
    margin_bottom = random_bottom_margin()
    margin_top = random_top_margin()

        
    lines = read_random_lines_from_file(Lines, number_of_lines_to_read)
    image_save_name = i

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    if not os.path.exists(line_box_save_path):
        os.makedirs(line_box_save_path)

    if not os.path.exists(word_box_save_path):
        os.makedirs(word_box_save_path)
    
    generate_text_image(image_save_name, image_save_path, lang, lines, background_image_path, line_box_save_path, 
                        word_box_save_path, blur, color_code, font_path, font_size, word_space, line_space,
                        margin, margin_bottom, margin_top)
