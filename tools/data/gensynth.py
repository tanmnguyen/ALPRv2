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


class EmptyBackgroundImageCaptcha(ImageCaptcha):
    # Override
    def generate_image(self, chars: str) -> Image:
        """
        Generate the image of the given characters with an empty background set to 0.
        Args:
            chars (str): Characters to generate the image.
        Returns:
            Image: Captcha image of the given characters.
        """
        background = 0, 0, 0, 0
        color = _random_color(100, 250, 255)
        im = self.create_captcha_image(chars, color, background)
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        im = im.filter(SMOOTH)
        return im


class MultiCaptchaImageGenerator:
    def __init__(
        self,
        height: int,
        width: int,
        text_tokens: List[str],
        max_tokens: int,
        max_lines: int,
    ):
        self.height = height
        self.width = width
        self.text_tokens = text_tokens
        self.max_tokens = max_tokens
        self.max_lines = max_lines

    def generate_captcha_texts(self):
        """
        Generate captcha text lines.
        The function generates a random selection of tokens and splits them into lines.
        Returns:
            list: List of text lines.
        """
        # Generate a random selection of tokens
        text = np.random.choice(self.text_tokens, np.random.randint(1, self.max_tokens))

        # Generate random split points without replacement and sort them
        split_points = sorted(
            np.random.choice(
                range(1, len(text)),
                np.random.randint(1, min(self.max_lines, len(text)) + 1) - 1,
                replace=False,
            )
        )

        # Split the text based on the split points
        text_lines = [
            "".join(text[start:end])
            for start, end in zip([0] + split_points, split_points + [len(text)])
        ]

        return text_lines

    def generate_captcha_attribute(self):
        """
        Generate captcha object attributes.
        Return:
            dict: Object attributes all in relative values:
                - x: x-coordinate of the object center.
                - y: y-coordinate of the object center.
                - w: width of the object.
                - h: height of the object.
                - rotation: rotation of the object.
                - content: content of the object. The content should be a list of string, where each element
                    corresponds to a line of text.
        """

        w = np.random.uniform(0.3, 0.5)
        h = np.random.uniform(0.3, 0.5)
        x = np.random.uniform(w / 2, 1 - w / 2)
        y = np.random.uniform(h / 2, 1 - h / 2)
        rotation = np.random.uniform(-50, 50)
        text_lines = self.generate_captcha_texts()

        return {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "rotation": rotation,
            "text_lines": text_lines,
        }

    def is_captcha_intersect(self, captcha_att_list: List[dict], captcha_att: dict):
        """
        Check if the captcha object intersects with any of the existing captcha objects before rotation.
        Args:
            captcha_att_list (list): List of captcha object attributes.
            captcha_att (dict): Captcha object attributes.
        Return:
            bool: True if the captcha object intersects with any of the existing captcha objects.
        """
        x = captcha_att["x"]
        y = captcha_att["y"]
        w = captcha_att["w"]
        h = captcha_att["h"]

        for att in captcha_att_list:
            x1 = att["x"]
            y1 = att["y"]
            w1 = att["w"]
            h1 = att["h"]

            if x < x1 + w1 and x + w > x1 and y < y1 + h1 and y + h > y1:
                return True

        return False

    def insert_captcha_object(self, img: np.ndarray, captcha_att: dict):
        """
        Inplace insert a captcha object into the image.
        The captcha attribute is used to rotate and insert the object into the image.
        After the insertion, the attribute is updated correspondingly.
        Args:
            img (np.ndarray): Image to insert the object into.
            captcha_att (dict): Captcha object attributes.
        """
        # Get the captcha object attributes
        x = int(captcha_att["x"] * img.shape[1])
        y = int(captcha_att["y"] * img.shape[0])
        w = int(captcha_att["w"] * img.shape[1])
        h = int(captcha_att["h"] * img.shape[0])
        rotation = captcha_att["rotation"]
        text_lines = captcha_att["text_lines"]

        img_list = []
        for text in text_lines:
            imageCaptcha = EmptyBackgroundImageCaptcha(height=h, width=w)
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
        img[
            y - h // 2 : y - h // 2 + h,
            x - w // 2 : x - w // 2 + w,
        ] += stacked_text

    def generate_multi_captcha_image(self, num_max_objs: int, add_noise: bool):
        """
        Generate an image with multiple captcha objects.
        The number of captcha objects is upper bounded by num_max_objs.

        Args:
            num_max_objs (int): Maximum number of captcha objects to generate.
            add_noise (bool): Whether to add noise to the image.
        Returns:
            np.ndarray: Image with multiple captcha objects.
            list: List of captcha object attributes.
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Add multiple captcha objects into the image.
        captcha_att_list = []
        for _ in range(num_max_objs):
            captcha_att = self.generate_captcha_attribute()

            if not self.is_captcha_intersect(captcha_att_list, captcha_att):
                self.insert_captcha_object(img, captcha_att)
                captcha_att_list.append(captcha_att)

        img = cv2.bitwise_not(img)

        # 2. Add noise to the image.
        if add_noise:
            noise = np.random.randint(0, 255, img.shape, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)

        return img, captcha_att_list


def gen_and_save_image(
    image_idx: int,
    num_max_objs: int,
    multicaptcha_image_generator: MultiCaptchaImageGenerator,
    result_dir: str,
):
    """
    Generate and save an image and its labels.
    Args:
        image_idx (int): Index of the image.
        num_max_objs (int): Maximum number of captcha objects to generate.
        multicaptcha_image_generator (MultiCaptchaImageGenerator): MultiCaptchaImageGenerator instance.
        result_dir (str): Path to the directory to save the images.
    """
    # 1. generate the image
    img, att_list = multicaptcha_image_generator.generate_multi_captcha_image(
        num_max_objs=num_max_objs, add_noise=True
    )

    # 2. save the image
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    img_path = os.path.join(result_dir, f"{image_idx}.jpg")  # jpg for compression
    cv2.imwrite(img_path, img)

    # 3. save the labels as YOLO format
    label_path = os.path.join(result_dir, f"{image_idx}.txt")
    with open(label_path, "w") as f:
        for att in att_list:
            x = att["x"]
            y = att["y"]
            w = att["w"]
            h = att["h"]
            text_lines = att["text_lines"]
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {''.join(text_lines)}\n")


def main(args):
    spec = parse_spec(args.spec)

    multicaptcha_image_generator = MultiCaptchaImageGenerator(
        height=int(spec["imgh"]),
        width=int(spec["imgw"]),
        text_tokens=list(spec["text_tokens"]),
        max_tokens=int(spec["max_tokens"]),
        max_lines=args.max_lines,
    )

    num_data = {
        "train": args.num_train,
        "val": args.num_val,
    }

    for partition, num in num_data.items():
        print(f"Generating {num} synthetic images for {partition}...")
        for i in tqdm(range(num)):
            gen_and_save_image(
                image_idx=i,
                num_max_objs=args.max_objs,
                multicaptcha_image_generator=multicaptcha_image_generator,
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
