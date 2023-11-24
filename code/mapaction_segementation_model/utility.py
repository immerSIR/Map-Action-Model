import os
import torch

from torchvision.io import read_image, ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class MapActionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, classes={1: "D-Solide"}):
        self.root = root
        self.transforms = transforms
        self.classes = classes  # Dictionary mapping class names to integer labels

        self.imgs = []
        self.masks = []

        for class_label, class_name in self.classes.items():
            images_path = os.path.join(root, "images2", class_name)
            masks_path = os.path.join(root, "Mask", class_name)

            class_imgs = list(sorted(os.listdir(images_path)))
            class_masks = list(sorted(os.listdir(masks_path)))

            self.imgs.extend([(class_label, img) for img in class_imgs])
            self.masks.extend([(class_label, mask) for mask in class_masks])

    def __getitem__(self, idx):
        class_label, img_name = self.imgs[idx]
        mask_name = self.masks[idx][1]

        img_path = os.path.join(self.root, "images2", self.classes[class_label], img_name)
        mask_path = os.path.join(self.root, "Mask", self.classes[class_label], mask_name)

        img = read_image(img_path, mode=ImageReadMode.RGB)
        mask = read_image(mask_path)

        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        boxes = masks_to_boxes(masks)

        boxes = []
        for mask in masks:
            pos = torch.where(mask)
            if pos[0].numel() == 0 or pos[1].numel() == 0:
                # Skip masks with no positive pixels
                continue
            xmin = torch.min(pos[1]).item()
            xmax = torch.max(pos[1]).item()
            ymin = torch.min(pos[0]).item()
            ymax = torch.max(pos[0]).item()
            # Ensure positive height and width
            if xmin == xmax or ymin == ymax:
                continue
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes)
        image_id = idx
        img = tv_tensors.Image(img)
        labels = (torch.ones((num_objs,), dtype=torch.int64) * class_label)
        #print(labels)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="xyxy", canvas_size=F.get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
