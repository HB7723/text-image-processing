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
    """
    Extract Hindi words from a given text line.

    Args:
    - line (str): Text line from which Hindi words will be extracted.

    Returns:
    - list: List of Hindi words found in the line.
    """
    hindi_words = re.findall(r'\b[\u0900-\u097F]+\b', line)
    return hindi_words


def extract_assamese_words(line):
    """
    Extract Assamese words from a given text line.

    Args:
    - line (str): Text line from which Assamese words will be extracted.

    Returns:
    - list: List of Assamese words found in the line.
    """
    assamese_words = re.findall(r'\b[\u0980-\u09FF]+\b', line)
    return assamese_words


def extract_english_words(line):
    """
    Extract English words from a given text line.

    Args:
    - line (str): Text line from which English words will be extracted.

    Returns:
    - list: List of English words found in the line.
    """
    english_words = re.findall(r'\b[a-zA-Z]+\b', line)
    return english_words


def thin_pg_blur_img(image, avg_height, std_dev):
    """
    Apply a Gaussian blur to an image with a dynamically calculated kernel size based on average height.

    Args:
    - image (Image.Image): The input image to be blurred.
    - avg_height (int): Average height of text lines used to determine the kernel size.
    - std_dev (float): Standard deviation for the Gaussian kernel.

    Returns:
    - Image.Image: The blurred image.
    """

    image_np = np.array(image)
    sigma = std_dev

    blurred_image_np = cv2.GaussianBlur(
        image_np, (2 * (avg_height) + 1, 4 * (avg_height) + 1), sigma)

    image_size = (2 * (avg_height) + 1, 2 *
                  (avg_height) + 1)

    gaussian_kernel = cv2.getGaussianKernel(image_size[0], sigma)

    plt.figure(figsize=(8, 4))
    plt.plot(gaussian_kernel)
    plt.title('Gaussian Window (cv2.GaussianBlur)')
    plt.xlabel('Pixel')
    plt.ylabel('Amplitude')
    plt.grid(True)

    blurred_image = Image.fromarray(blurred_image_np)
    print(avg_height)

    return blurred_image


def blur_image(image, blur_level):
    """
    Apply a simple Gaussian blur to an image.

    Args:
    - image (Image.Image): The input image to be blurred.
    - blur_level (int): Intensity of the blur.

    Returns:
    - Image.Image: The blurred image.
    """
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_level))

    return blurred_image


def binarize_img(image, threshold=220):
    """
    Convert an image to binary (black and white) using a threshold.

    Args:
    - image (Image.Image): The image to be binarized.
    - threshold (int): Pixel intensity threshold used for binarization.

    Returns:
    - Image.Image: The binarized image.
    """
    image_np = np.array(image)

    gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    _, binarized_image_np = cv2.threshold(
        gray_image_np, threshold, 255, cv2.THRESH_BINARY)

    binarized_image = Image.fromarray(binarized_image_np)

    return binarized_image


def get_random_page_content() -> str:
    """
    Retrieve the summary content of a random Wikipedia page.

    Returns:
    - str: Summary content of the Wikipedia page.
    """
    page_title = wikipedia.random(1)
    try:
        page_content = wikipedia.page(page_title).summary
    except (wikipedia.DisambiguationError, wikipedia.PageError):
        return get_random_page_content()
    return page_content


def create_strings_from_wikipedia(lang: str, minimum_length: int, count: int) -> list:
    """
    Generate a list of strings from random Wikipedia articles that exceed a minimum length.

    Args:
    - lang (str): Language setting for Wikipedia.
    - minimum_length (int): Minimum length of sentences to consider.
    - count (int): Number of sentences to retrieve.

    Returns:
    - list: List of strings (sentences) that meet the criteria.
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
    Read a specified number of random lines from a text file.

    Args:
    - filename (str): Path to the file from which lines will be read.
    - num_lines (int): Number of random lines to read.

    Returns:
    - list: List of randomly selected lines from the file.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    selected_lines = random.sample(lines, num_lines)
    return selected_lines


def generate_word_bbox(image_save_name: str, word_box_save_path: str, words_with_bboxes: list) -> None:
    """
    Write word bounding boxes to a text file in a specified directory.

    Args:
    - image_save_name (str): Base name for the output file.
    - word_box_save_path (str): Directory path to save the output file.
    - words_with_bboxes (list): List of tuples containing words and their bounding boxes.
    """

    output_file = os.path.join(word_box_save_path, f"{image_save_name}.txt")

    with open(output_file, 'w') as f:
        for word, bbox in words_with_bboxes:
            bbox_str = ','.join(str(coord) for coord in bbox)
            f.write(f"{bbox_str}    {word}\n")


def generate_line_bbox(image_save_name: str, line_box_save_path: str, lines_with_bboxes: list) -> None:
    """
    Write line bounding boxes to a text file in a specified directory.

    Args:
    - image_save_name (str): Base name for the output file.
    - line_box_save_path (str): Directory path to save the output file.
    - lines_with_bboxes (list): List of tuples containing lines of text and their bounding boxes.
    """

    output_file = os.path.join(line_box_save_path, f"{image_save_name}.txt")

    with open(output_file, 'w') as f:
        for line, line_bbox in lines_with_bboxes:
            line_bbox_str = ','.join(str(coord) for coord in line_bbox)
            f.write(f"{line_bbox_str}\n")


def generate_text_image(image_save_name: str, gen_folder_path: str, lang: str, lines: list, background_image_path: str, line_box_save_path: str, word_box_save_path: str,
                        blur: int, color_code: tuple, font_path: str, font_size: int, word_space: int, line_space: int, margin: int, margin_bottom: int, margin_top: int, image_save_path: str) -> list:
    """
    Generate an image with overlaid text from a list of lines and save it along with bounding boxes for words and lines.

    Args:
    - image_save_name (str): Name to be used for saving the image.
    - gen_folder_path (str): Path to the general folder for generating files.
    - lang (str): Language of the text.
    - lines (list): List of text lines to be rendered on the image.
    - background_image_path (str): Path to the background image.
    - line_box_save_path (str): Path to save line bounding boxes.
    - word_box_save_path (str): Path to save word bounding boxes.
    - blur (int): Blur intensity for the final image.
    - color_code (tuple): Color code for the text.
    - font_path (str): Path to the font file.
    - font_size (int): Font size.
    - word_space (int): Space between words.
    - line_space (int): Space between lines.
    - margin (int): Margin around the text in the image.
    - margin_bottom (int): Bottom margin.
    - margin_top (int): Top margin.
    - image_save_path (str): Directory to save the final image.
    """

    img_width, img_height = 600, 800
    max_word_height = 0
    max_text_width = img_width - margin
    max_text_height = img_height - margin_top

    img = Image.open(background_image_path).resize(
        (img_width, img_height))
    img_white_bg = Image.new('RGB', (img_width, img_height), color='white')

    draw = ImageDraw.Draw(img)
    draw_white_bg = ImageDraw.Draw(img_white_bg)

    font = ImageFont.truetype(
        font_path if font_path else ImageFont.load_default(), font_size)

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
                lines_with_bboxes.append(
                    (final_line, (margin, y_start_per_word, x_start_per_word - word_space, y_start_per_word + max_word_height)))

                single_line = []
                line_width = 0
                x_start_per_word = margin
                y_start_per_word += (max_word_height + line_space)
                max_word_height = 0
                if y_start_per_word+margin_bottom > max_text_height:
                    break

            word_bbox = (x_start_per_word, y_start_per_word, x_start_per_word +
                         word_width, y_start_per_word + word_height)
            words_with_bboxes.append((word, word_bbox))
            single_line.append(word)

            draw.text((x_start_per_word, y_start_per_word), word,
                      0, font=font)
            draw_white_bg.text(
                (x_start_per_word, y_start_per_word), word, 0, font=font)

            line_width = line_width + (word_width + word_space)
            x_start_per_word = x_start_per_word + (word_width + word_space)

    final_line = ' '.join(char for char in single_line)
    lines_with_bboxes.append((final_line, (margin, y_start_per_word,
                             x_start_per_word - word_space, y_start_per_word + max_word_height)))

    img = blur_image(img, blur)
    save_image_with_unique_name(image_save_name, image_save_path, img)

    generate_line_bbox(image_save_name, line_box_save_path, lines_with_bboxes)
    generate_word_bbox(image_save_name, word_box_save_path, words_with_bboxes)
    save_image_with_unique_name(image_save_name, image_save_path, img)


def save_image_with_unique_name(image_save_name, image_save_path, image):
    """
    Save an image with a unique name in a specified directory.

    Args:
    - image_save_name (str): Name to be used for the image file.
    - image_save_path (str): Path where the image should be saved.
    - image (Image.Image): Image to be saved.
    """
    filename = os.path.join(image_save_path, f"{image_save_name}.png")
    image.save(filename)
    print(f"Image saved as: {filename}")


def choose_background_image():
    """
    Select a random background image from a predefined directory.

    Returns:
    - str: Path to the selected background image.
    """
    images_folder = 'background/'
    background_images = [os.path.join(images_folder, img) for img in os.listdir(
        images_folder) if img.endswith(('.jpeg', '.jpg', '.png'))]
    return background_images[random.randint(0, len(background_images) - 1)]


def choose_font(language: str):
    """
    Select a random font file based on the specified language from a predefined directory.

    Args:
    - language (str): Language setting to determine the directory of fonts.

    Returns:
    - str: Path to the selected font file.
    """
    fonts_folder = 'fonts/fonts/' + language
    fonts = [os.path.join(fonts_folder, f) for f in os.listdir(
        fonts_folder) if f.endswith(('.ttf', '.otf'))]
    return fonts[random.randint(0, len(fonts) - 1)]


def choose_text_color():
    """
    Generate a random text color in RGBA format.

    Returns:
    - tuple: Randomly generated text color.
    """
    return (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), 255)


def choose_blur_level():
    """
    Generate a random blur level.

    Returns:
    - float: Randomly determined blur intensity.
    """
    return random.uniform(0.5, 1)


def choose_language():
    """
    Randomly select a language from a predefined list.

    Returns:
    - str: Selected language.
    """
    languages = ["en"]
    return random.choice(languages)


def choose_text_size():
    """
    Randomly determine the text size.

    Returns:
    - int: Randomly chosen text size.
    """
    return random.randint(5, 20)


def choose_word_space():
    """
    Randomly determine the space between words.

    Returns:
    - int: Randomly chosen space between words.
    """
    return random.randint(3, 10)


def choose_line_space():
    """
    Randomly determine the space between lines.

    Returns:
    - int: Randomly chosen space between lines.
    """
    return random.randint(6, 10)


def choose_lines_read():
    """
    Randomly determine the number of lines to read.

    Returns:
    - int: Randomly chosen number of lines to read.
    """
    return random.randint(1, 200)


def choose_margin():
    """
    Randomly determine the margin size around the text.

    Returns:
    - int: Randomly chosen margin size.
    """
    return random.randint(6, 20)


def random_bottom_margin():
    """
    Randomly determine the bottom margin size.

    Returns:
    - int: Randomly chosen bottom margin size.
    """
    return random.randint(6, 20)


def random_top_margin():
    """
    Randomly determine the top margin size.

    Returns:
    - int: Randomly chosen top margin size.
    """
    return random.randint(6, 20)


def alter_image(background_image_path):
    """
    Increase the intensity of all pixels in the background image.

    Args:
    - background_image_path (str): Path to the background image file.

    Returns:
    - np.ndarray: Modified image with increased pixel intensity.
    """
    image = cv2.imread(background_image_path)

    if image is None:
        print("Error: Could not open or find the image.")
        exit()

    increase_value = 50

    increased_image = np.clip(image.astype(
        np.int32) + increase_value, 0, 255).astype(np.uint8)

    return increased_image


def main():
    print("Automated Custom Text Image Generator Running...")
    lang = 'en'
    color_code = (255, 255, 255, 255)
    background_image_path = choose_background_image()
    font_path = choose_font(lang)
    font_size = 18
    word_space = 10
    line_space = 15
    number_of_lines_to_read = 50
    margin = 20
    margin_bottom = 30
    margin_top = 30
    blur = 0.5

    image_save_path = 'output/images'
    line_box_save_path = 'output/line_bbox'
    word_box_save_path = 'output/word_bbox'
    Lines = "text/text/file/with/words.txt"

    for path in [image_save_path, line_box_save_path, word_box_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    lines = read_random_lines_from_file(Lines, number_of_lines_to_read)
    image_save_name = "custom_image"

    generate_text_image(image_save_name, image_save_path, lang, lines, background_image_path, line_box_save_path,
                        word_box_save_path, blur, color_code, font_path, font_size, word_space, line_space,
                        margin, margin_bottom, margin_top, image_save_path)

    print(
        f"Image '{image_save_name}.png' generated successfully in '{image_save_path}'.")


if __name__ == "__main__":
    main()
