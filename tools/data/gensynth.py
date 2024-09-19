import sys

sys.path.append("../")

import os
import cv2
import random
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from sdk.fileio import parse_spec
from PIL.ImageFilter import SMOOTH
from captcha.image import ImageCaptcha
from typing import List, Union, Tuple, Optional


ColorTuple = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


def _random_color(start: int, end: int, opacity: Optional[int]) -> ColorTuple:
    """
    Generate a random color. Code taken from
    https://github.com/lepture/captcha/blob/master/src/captcha/image.py#L196

    Args:
        start (int): Start value for the color.
        end (int): End value for the color.
        opacity (int, optional): Opacity value for the color.

    Returns:
        ColorTuple: Tuple of color values.
            The tuple will have the values in the order (red, green, blue, opacity).
            If opacity is None, the tuple will have 3 values, else 4 values.
    """
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return red, green, blue
    return red, green, blue, opacity


class ModifiedImageCaptcha(ImageCaptcha):
    # Override
    def generate_image(self, chars: str) -> Image:
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = 0, 0, 0, 0
        color = _random_color(100, 250, 255)
        im = self.create_captcha_image(chars, color, background)
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        im = im.filter(SMOOTH)
        return im


def generate_text_lines(text_tokens, max_tokens, max_lines):
    """
    Generate text lines. The function generates a random selection of tokens and splits them into lines.
    Args:
        text_tokens (list): List of text tokens.
        max_tokens (int): Maximum number of tokens.
        max_lines (int): Maximum number of lines of text
    Returns:
        list: List of text lines.
    """
    # Generate a random selection of tokens
    text = np.random.choice(text_tokens, np.random.randint(1, max_tokens))

    # Generate random split points without replacement and sort them
    split_points = sorted(
        np.random.choice(
            range(1, len(text)),
            np.random.randint(1, min(max_lines, len(text)) + 1) - 1,
            replace=False,
        )
    )

    # Split the text based on the split points
    content = [
        "".join(text[start:end])
        for start, end in zip([0] + split_points, split_points + [len(text)])
    ]

    return content


def insert_object(img, obj_attr):
    """
    Inplace insert an object into the image.
    Args:
        img (np.ndarray): Image to insert the object into.
        obj_attr (dict): Object attributes.

    """
    # Get the object attributes
    x = int(obj_attr["x"] * img.shape[1])
    y = int(obj_attr["y"] * img.shape[0])
    w = int(obj_attr["w"] * img.shape[1])
    h = int(obj_attr["h"] * img.shape[0])
    rotation = obj_attr["rotation"]
    content = obj_attr["content"]

    img_list = []
    for text in content:
        imageCaptcha = ModifiedImageCaptcha(height=h, width=w)
        text_img = imageCaptcha.generate_image(text)
        img_list.append(text_img)

    # stack the image list vertically
    stacked_text = np.vstack(img_list)

    # add a border line to the stacked text
    stacked_text = cv2.copyMakeBorder(
        stacked_text, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(100, 100, 100)
    )

    # Rotate the image and expand it to fit the new size, padding with white
    stacked_text = Image.fromarray(stacked_text)
    stacked_text = np.array(
        stacked_text.rotate(rotation, expand=True, fillcolor=(0, 0, 0))
    )

    # resize the image to the object size
    stacked_text = cv2.resize(stacked_text, (w, h))

    # Insert the rotated text into the image
    img[y : y + stacked_text.shape[0], x : x + stacked_text.shape[1]] += stacked_text


def generate_object_attribute(text_tokens: List[str], max_tokens: int, max_lines: int):
    """
    Generate object attributes.

    Args:
        text_tokens (list): List of text tokens.
        max_tokens (int): Maximum number of tokens.
        max_lines (int): Maximum number of lines of text
    Return:
        dict: Object attributes all in relative values:
            - x: x-coordinate of the object top left.
            - y: y-coordinate of the object top left.
            - w: width of the object.
            - h: height of the object.
            - rotation: rotation of the object.
            - content: content of the object. The content should be a list of string, where each element
                corresponds to a line of text.
    """

    w = np.random.uniform(0.3, 0.5)
    h = np.random.uniform(0.3, 0.5)
    x = np.random.uniform(0, 1 - w)
    y = np.random.uniform(0, 1 - h)
    rotation = np.random.uniform(0, 80)
    content = generate_text_lines(text_tokens, max_tokens, max_lines)

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "rotation": rotation,
        "content": content,
    }


def att_intersect(att_list, obj_attr):
    """
    Check if the object intersects with any of the existing objects.

    Args:
        att_list (list): List of object attributes.
        obj_attr (dict): Object attributes.

    Returns:
            bool: True if the object intersects with any of the existing objects.
    """
    x = obj_attr["x"]
    y = obj_attr["y"]
    w = obj_attr["w"]
    h = obj_attr["h"]

    for att in att_list:
        x1 = att["x"]
        y1 = att["y"]
        w1 = att["w"]
        h1 = att["h"]

        if x < x1 + w1 and x + w > x1 and y < y1 + h1 and y + h > y1:
            return True

    return False


def gen_and_save_image(image_idx, text_tokens, max_tokens, height, width, result_dir):
    """
    Generate and save an image and its labels.
    Args:
        image_idx (int): Index of the image.
        text_tokens (list): List of text tokens.
        max_tokens (int): Maximum number of tokens.
        height (int): Height of the image.
        width (int): Width of the image.
        result_dir (str): Path to the directory to save the
            synthetic images.
    """
    # 1. generate a random number of objects
    att_list = []
    for _ in range(args.max_objs):
        obj_attr = generate_object_attribute(text_tokens, max_tokens, args.max_lines)

        if not att_intersect(att_list, obj_attr):
            att_list.append(obj_attr)

    # 2. insert the objects into the image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for att in att_list:
        insert_object(img, att)
    img = cv2.bitwise_not(img)

    # 3. add noise
    noise = np.random.randint(0, 255, img.shape, dtype=np.uint8)
    img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)

    # 4. save the image
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    img_path = os.path.join(result_dir, f"{image_idx}.jpg")  # jpg for compression
    cv2.imwrite(img_path, img)

    # 5. save the labels as YOLO format
    label_path = os.path.join(result_dir, f"{image_idx}.txt")
    with open(label_path, "w") as f:
        for att in att_list:
            x = att["x"] + att["w"] / 2
            y = att["y"] + att["h"] / 2
            w = att["w"]
            h = att["h"]
            content = att["content"]
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {''.join(content)}\n")


def main(args):
    spec = parse_spec(args.spec)

    text_tokens = list(spec["text_tokens"])
    max_tokens = int(spec["max_tokens"])
    height = int(spec["imgh"])
    width = int(spec["imgw"])

    num_data = {
        "train": args.num_train,
        "val": args.num_val,
    }

    for partition, num in num_data.items():
        print(f"Generating {num} synthetic images for {partition}...")
        for i in tqdm(range(num)):
            gen_and_save_image(
                image_idx=i,
                text_tokens=text_tokens,
                max_tokens=max_tokens,
                height=height,
                width=width,
                result_dir=os.path.join(args.result_dir, partition),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Data")

    parser.add_argument(
        "-spec", "--spec", required=True, type=str, help="Path to spec file (yaml)."
    )

    parser.add_argument(
        "-max-objs",
        "--max-objs",
        required=False,
        default=4,
        type=int,
        help="Number of maximum objects per image.",
    )

    parser.add_argument(
        "-max-lines",
        "--max-lines",
        required=False,
        default=3,
        type=int,
        help="Number of maximum text lines per image.",
    )

    parser.add_argument(
        "-num-train",
        "--num-train",
        required=False,
        default=2000,
        type=int,
        help="Number of generating synthetic images for training.",
    )

    parser.add_argument(
        "-num-val",
        "--num-val",
        required=False,
        default=500,
        type=int,
        help="Number of generating synthetic images for validation.",
    )

    parser.add_argument(
        "-result-dir",
        "--result-dir",
        required=False,
        default="results/synth-data-output",
        type=str,
        help="Path to the directory to save the synthetic images.",
    )

    args = parser.parse_args()
    main(args)
