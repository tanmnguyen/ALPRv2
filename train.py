import torch
import argparse
from models.ALPRv2 import ALPRv2, Trainer
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionPredictor


def main(args):
    # in_height, in_width = 512, 512
    # model = ALPRv2("yolov10n.yaml", in_height, in_width)
    # # model.init_criterion()

    # inp = torch.rand(2, 3, 512, 512)
    # bbox = torch.rand(2, 4)  # x, y, w, h (normalized)
    # out, y = model(x=inp, bbox=bbox)

    # print(y.shape)
    # predictor
    # detector = DetectionPredictor()
    # output = detector(torch.rand(1, 3, 512, 512), model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for ALPRv2")
    # parser.add_argument(
    #     "-images", "--images", required=True, help="Path to images folder."
    # )

    args = parser.parse_args()
    main(args)
