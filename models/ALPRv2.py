# STILL IN DEVELOPMENT. CODE IS NEEDED TO BE CLEANED UP AND TESTED.
# NOT PRODUCTION READY !!!
# AUTHOR: TAN NGUYEN (2024) - nguyenmanhtan.02@gmail.com - tmn2134@columbia.edu

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

import torch
import torch.nn as nn
import traceback
import einops

from .HookManager import FeaturesExtractorHookManager


class SimpleFPN(nn.Module):
    def __init__(self, feature_dims, hidden_channels=128, out_channels=256):
        super().__init__()

        self.convs = nn.ModuleList()
        for i, dim in enumerate(feature_dims):
            self.convs.add_module(
                f"channel_scaler_conv_{i}", nn.Conv2d(dim[1], hidden_channels, 1)
            )

        self.deconvs = nn.ModuleDict()
        for i in range(1, len(feature_dims)):
            prv_dim, cur_dim = feature_dims[i - 1], feature_dims[i]
            if prv_dim[2:] == cur_dim[2:]:
                module = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
            elif prv_dim[2] < cur_dim[2]:
                module = nn.Conv2d(hidden_channels, hidden_channels, 3, 2, 1)
            else:
                module = nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, 2)

            self.deconvs.add_module(
                f"upsample_{i}",
                nn.Sequential(
                    module,
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                ),
            )

        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)

    def forward(self, features):
        # Scale the channels
        channel_scaled = []
        for i, ch_conv in enumerate(self.convs):
            channel_scaled.append(ch_conv(features[i]))

        # Upsample the features
        out = channel_scaled[-1]
        for i in range(len(channel_scaled) - 2, -1, -1):
            out = self.deconvs[f"upsample_{i + 1}"](out) + channel_scaled[i]

        out = self.out_conv(out)
        return out


class TextDecoder(nn.Module):
    def __init__(
        self,
        hidden_channels=256,
        decoder_heads=8,
        text_decoder_layers=1,
        max_seq_len=10,
        num_classes=32,
    ):
        """
        Initialize the text decoder.
        Args:
            hidden_channels (int): hidden channels for the model.
            decoder_heads (int): number of heads for the transformer decoder.
            text_decoder_layers (int): number of layers for the transformer decoder.
            max_seq_len (int): maximum sequence length for the transformer decoder.
            num_classes (int): number of classes for the transformer
        """
        super().__init__()

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_channels, nhead=decoder_heads),
            num_layers=text_decoder_layers,
        )

        self.indices = nn.Parameter(torch.rand(max_seq_len, 1, hidden_channels))

        self.linear = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        """
        Decode the feature into text logits.
        Args:
            x (tensor): (B, N, C, H, W)

        Returns:
            tensor: (B, N, max_seq_len, num_classes). This is the logits for the text.
        """

        B, N, C, H, W = x.shape
        x = einops.rearrange(x, "b n c h w -> (h w) (b n) c")

        # decode the feature
        indices = self.indices.repeat(1, B * N, 1)
        x = self.decoder(indices, x)  # (max_seq_len, B * N, C)

        assert len(x.shape) == 3, f"Invalid shape: {x.shape}"
        assert x.shape[0] == indices.shape[0], f"Invalid shape: {x.shape}"

        # reshape the feature
        x = einops.rearrange(
            x, "s (b n) c -> b n s c", b=B, n=N
        )  # (B, N, max_seq_len, C)

        # decode the feature into text
        x = self.linear(x)

        return x


class JointDetectionTextRecognitionLoss:
    """
    Criterion for the joint detection and text recognition loss
    """

    def __init__(self, detection_criterion):
        """
        Initialize the criterion.
        Args:
            detection_criterion: detection criterion.
        """
        self.detection_criterion = detection_criterion
        self.text_criterion = nn.CrossEntropyLoss()

    def __call__(self, det_preds, txt_preds, batch):
        """
        Compute the loss.
        Args:
            det_preds: detection predictions.
            det_batch: detection batch.
            txt_preds: text predictions.
            txt_batch: text batch.

        Returns:
            tuple: loss and loss items.
        """
        det_loss, det_loss_items = self.detection_criterion(det_preds, det_batch)
        txt_loss = self.text_criterion(txt_preds, txt_batch["txt"])

        loss = det_loss + txt_loss
        loss_items = det_loss_items

        return loss, loss_items


class ALPRv2(DetectionModel):
    def __init__(
        self,
        cfg,
        in_height,
        in_width,
        fpn_hidden_channels=128,
        hidden_channels=256,
        decoder_heads=8,
        text_decoder_layers=1,
        lp_sz=(64, 64),
        max_seq_len=10,
        num_classes=32,
    ):
        """
        Initialize the ALPRv2 model.
        Args:
            cfg (str): path to the model configuration file.
            in_height (int): input image height.
            in_width (int): input image width.
            lp_sz (tuple): license plate size (height, width).
            fpn_hidden_channels: hidden channels for the FPN.
            hidden_channels: hidden channels for the model.
            decoder_heads: number of heads for the transformer decoder.
            text_decoder_layers: number of layers for the transformer decoder.
            max_seq_len: maximum sequence length for the transformer decoder.
            num_classes: number of classes for the transformer decoder.
        """
        super().__init__(cfg)
        self.lp_sz = lp_sz

        # hook manager to extract features at different layers
        self.features_extractor = FeaturesExtractorHookManager(self)

        # activate the hook to get features output
        dummy = torch.rand(1, 3, in_height, in_width)
        _ = super().forward(dummy)

        self.simple_fpn = SimpleFPN(
            feature_dims=[f.shape for f in self.features_extractor.get_layer_output()],
            hidden_channels=fpn_hidden_channels,
        )

        self.text_decoder = TextDecoder(
            hidden_channels=hidden_channels,
            decoder_heads=decoder_heads,
            text_decoder_layers=text_decoder_layers,
            max_seq_len=max_seq_len,
            num_classes=num_classes,
        )

    def forward(self, x, bbox=None, **kwargs):
        """Forward pass of the model.

        Args:
            x (tensor): input tensor.
            bbox (list): list of bounding boxes. Each bounding box is a tuple (x, y, w, h).
                The coordinates are normalized into the range [0, 1] relative to the input image size.
        """
        yolo_out = super().forward(x, **kwargs)

        if bbox is None:
            return yolo_out

        features_list = self.features_extractor.get_layer_output()
        fpn = self.simple_fpn(features_list)

        _, _, H, W = fpn.shape

        # extract feature in the bounding box region
        crops = []
        for box in bbox:
            x, y, w, h = box  # x, y, w, h (normalized)
            x1, y1, x2, y2 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)

            crops.append(fpn[..., y1:y2, x1:x2])

        # resize the etracted feature while keeping the aspect ratio
        for i, crop in enumerate(crops):
            h, w = crop.shape[-2:]
            if h > w:
                new_h = self.lp_sz[0]
                new_w = int((w / h) * new_h)
            else:
                new_w = self.lp_sz[1]
                new_h = int((h / w) * new_w)

            crops[i] = nn.functional.interpolate(
                crop, size=(new_h, new_w), mode="bilinear"
            )

        # pad the rest of the crop with zeros
        for i, crop in enumerate(crops):
            h, w = crop.shape[-2:]
            pad_h = self.lp_sz[0] - h
            pad_w = self.lp_sz[1] - w
            crops[i] = nn.functional.pad(
                crop, (0, pad_w, 0, pad_h)
            )  # default mode is zero padding

        crops = torch.stack(crops)  # (B, N, C, H, W)
        assert len(crops.shape) == 5, f"Invalid shape: {crops.shape}"

        # decode the feature into text
        text = self.text_decoder(crops)
        return yolo_out, text

    # Override
    def loss(self, batch):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.

        Returns:
            Tuple[torch.Tensor, dict]: Loss and loss items.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        det_pred, txt_pred = self.forward(batch["img"])
        return self.criterion(det_pred, txt_pred, batch)

    def init_criterion(self):
        """
        Initialize the criterion as a joint detection and text recognition loss.
        """
        criterion = JointDetectionTextRecognitionLoss(super().init_criterion())

        return criterion


class Trainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Returns a customized detection model instance configured with specified config and weights."""

        # init model 
        # TODO: specifies model parameters
        model = ALPRv2(cfg)

        # load weights 
        if weights:
            model.load(weights)

        return model
    

    # Override 
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataloader = super().get_dataloader(dataset_path, batch_size, rank, mode)

        # TODO: add label for license plate text
        return dataloader 

    