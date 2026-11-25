"""Data loading for EMCAF-Net. Label-safe online augmentation (masks are never noised or interpolated)."""
import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# ── Augmentation helpers ────────────────────────────────────────────────────

def cv_random_flip(img, label):
    """Random horizontal flip."""
    if random.randint(0, 1) == 1:
        img   = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    """Random crop within a 30-pixel border."""
    border = 30
    w, h   = image.size
    crop_w = np.random.randint(w - border, w)
    crop_h = np.random.randint(h - border, h)
    region = (
        (w - crop_w) >> 1, (h - crop_h) >> 1,
        (w + crop_w) >> 1, (h + crop_h) >> 1,
    )
    return image.crop(region), label.crop(region)


def randomRotation(image, label):
    """Random rotation in [−15°, 15°] applied with probability 0.2."""
    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)
        image = image.rotate(angle, Image.BICUBIC)
        label = label.rotate(angle, Image.NEAREST)
    return image, label


def colorEnhance(image):
    """Random brightness, contrast, colour and sharpness jitter."""
    image = ImageEnhance.Brightness(image).enhance(random.randint(5,  15) / 10.0)
    image = ImageEnhance.Contrast(image).enhance(  random.randint(5,  15) / 10.0)
    image = ImageEnhance.Color(image).enhance(     random.randint(0,  20) / 10.0)
    image = ImageEnhance.Sharpness(image).enhance( random.randint(0,  30) / 10.0)
    return image


def randomPeper(img):
    """Salt-and-pepper noise on the binary mask (0.15 % of pixels)."""
    img = np.array(img)
    n   = int(0.0015 * img.shape[0] * img.shape[1])
    for _ in range(n):
        rx = random.randint(0, img.shape[0] - 1)
        ry = random.randint(0, img.shape[1] - 1)
        img[rx, ry] = 0 if random.randint(0, 1) == 0 else 255
    return Image.fromarray(img)


# ── File pairing ────────────────────────────────────────────────────────────

IMG_EXTS = ('.tif', '.tiff', '.png', '.jpg')


def paired_files(image_root, gt_root):
    """Pair each image with the mask that has the SAME file stem.
    Raises an error if any image has no mask (instead of silently mis-pairing)."""
    def by_stem(root):
        return {os.path.splitext(f)[0]: os.path.join(root, f)
                for f in os.listdir(root) if f.lower().endswith(IMG_EXTS)}
    imgs, gts = by_stem(image_root), by_stem(gt_root)
    missing = sorted(set(imgs) - set(gts))
    if missing:
        raise FileNotFoundError(f'{len(missing)} image(s) in {image_root} have no mask in '
                                f'{gt_root}, e.g. {missing[:3]}')
    stems = sorted(imgs)
    return [imgs[s] for s in stems], [gts[s] for s in stems]


def mask_to_binary(mask_pil):
    """Any non-zero pixel = nucleus. Works for 0/1 and 0/255 masks."""
    return (np.array(mask_pil) > 0).astype(np.float32)


# ── Training / Validation dataset ───────────────────────────────────────────

class SalObjDataset(data.Dataset):
    """
    Dataset for training and validation.
    Set augment=True for training, augment=False for validation.
    """
    def __init__(self, image_root, gt_root, trainsize, augment=True):
        self.trainsize = trainsize
        self.augment   = augment

        self.images, self.gts = paired_files(image_root, gt_root)
        self.filter_files()
        self.size = len(self.images)
        print(f"[{image_root}] {self.size} image/mask pairs, augment={augment}")

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_resize = transforms.Resize((self.trainsize, self.trainsize),
                                           interpolation=transforms.InterpolationMode.NEAREST)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt    = self.binary_loader(self.gts[index])

        if self.augment:
            image, gt = cv_random_flip(image, gt)
            image, gt = randomCrop(image, gt)
            image, gt = randomRotation(image, gt)
            image     = colorEnhance(image)
            # randomPeper(gt) removed: it flipped 0.15% of LABEL pixels (label noise)

        gt = torch.from_numpy(mask_to_binary(self.gt_resize(gt))).unsqueeze(0)
        return self.img_transform(image), gt

    def filter_files(self):
        assert len(self.images) == len(self.gts), \
            f"Mismatch: {len(self.images)} images vs {len(self.gts)} masks"
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            if Image.open(img_path).size == Image.open(gt_path).size:
                images.append(img_path)
                gts.append(gt_path)
        dropped = len(self.images) - len(images)
        if dropped:
            print(f'WARNING: {dropped} pair(s) dropped because image and mask sizes differ')
        self.images, self.gts = images, gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


def _seed_worker(worker_id):
    """Give every DataLoader worker its own random/numpy seed (derived from torch)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_loader(image_root, gt_root, batchsize, trainsize, augment=True,
               shuffle=True, num_workers=4, pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, trainsize, augment=augment)
    return data.DataLoader(dataset=dataset,
                           batch_size=batchsize,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           worker_init_fn=_seed_worker,
                           persistent_workers=num_workers > 0)


# ── Test dataset (no augmentation, sequential) ──────────────────────────────

class test_dataset:
    """Sequential loader for inference. No augmentation applied."""
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images, self.gts = paired_files(image_root, gt_root)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size  = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        gt    = self.binary_loader(self.gts[self.index])
        name  = os.path.basename(self.images[self.index])
        if name.endswith('.jpg'):
            name = name.replace('.jpg', '.png')
        self.index = (self.index + 1) % self.size
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')