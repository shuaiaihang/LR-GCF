import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import get_image_size
from numpy import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torchvision.transforms import transforms as T
from torchvision.transforms.functional import InterpolationMode

def build_transforms(transforms_):
    transforms = []
    for transform in transforms_:
        if hasattr(transform, 'kwargs') and transform.kwargs is not None:
            kwargs = transform.kwargs.__dict__
            transform = eval(f"{transform.name}")(**kwargs)
        else:
            transform = eval(f"{transform.name}()")
        transforms.append(transform)
    return Compose(transforms)


class Normalize(T.Normalize):

    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)

    def forward(self, sample):
        sample['image'] = F.normalize(sample['image'], self.mean, self.std, self.inplace)
        return sample


class GaussianBlur(T.GaussianBlur):

    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=0.5):
        super().__init__(kernel_size, sigma)
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            sample['image'] = F.gaussian_blur(sample['image'], self.kernel_size, [sigma, sigma])
        return sample
class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, **kwargs):
        for t in self.transforms:
            img = t(img, **kwargs)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given sample randomly with a given probability."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, label = sample['image'], sample['label']
            sample['image'] = F.hflip(img)
            sample['label'] = F.hflip(label)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given sample randomly with a given probability."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, label = sample['image'], sample['label']
            sample['image'] = F.vflip(img)
            sample['label'] = F.vflip(label)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class RandomGenerator(object):
    def __init__(self, output_size, p_flip=0.5, p_rot=0.5):
        self.output_size = output_size
        self.p_flip = p_flip
        self.p_rot = p_rot

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 使用随机数判断是否应用旋转翻转或旋转
        if random.random() > self.p_flip:
            image, label = self.random_rot_flip(image, label)
        elif random.random() > self.p_rot:
            image, label = self.random_rotate(image, label)
        
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8)).unsqueeze(0)

        sample['image'] = image
        sample['label'] = label

        return sample
    
class RandomRotation180(torch.nn.Module):
    """随机旋转0/90/180/270度 - 适用于PIL Image和Tensor"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, label = sample['image'], sample['label']
            # 随机选择旋转角度 (0, 90, 180, 270度)
            angle =int( random.choice([0, 90, 180, 270]))
            if angle > 0:
                # F.rotate支持PIL Image和Tensor
                sample['image'] = F.rotate(img, angle, InterpolationMode.BILINEAR)
                sample['label'] = F.rotate(label, angle, InterpolationMode.NEAREST)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    @staticmethod
    def random_rot_flip(image, label):
        """旋转和翻转"""
        k = np.random.randint(0, 4)
        image = np.rot90(image, k).copy()
        label = np.rot90(label, k).copy()

        axis = np.random.randint(0, 1)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    @staticmethod
    def random_rotate(image, label):
        """随机角度旋转"""
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label

class ToRGB:

    def __call__(self, sample):
        if sample['image'].shape[0] == 1:
            sample['image'] = sample['image'].repeat(3, 1, 1)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomCrop(T.RandomCrop):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def forward(self, sample):
        img = sample['image']
        label = sample['label']
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        width, height =  get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        sample['image'] = F.crop(img, i, j, h, w)
        sample['label'] = F.crop(label, i, j, h, w)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class RandomFlip(torch.nn.Module):

    def __init__(self, p=0.5, direction='vertical'):
        super().__init__()
        assert 0 <= p <= 1
        assert direction in ['horizontal', 'vertical', None], 'direction should be horizontal, vertical or None'
        self.p = p
        self.direction = direction

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, label = sample['image'], sample['label']
            if self.direction == 'horizontal':
                sample['image'] = F.hflip(img)
                sample['label'] = F.hflip(label)
            elif self.direction == 'vertical':
                sample['image'] = F.vflip(img)
                sample['label'] = F.vflip(label)
            else:
                if torch.rand(1) < 0.5:
                    sample['image'] = F.hflip(img)
                    sample['label'] = F.hflip(label)
                else:
                    sample['image'] = F.vflip(img)
                    sample['label'] = F.vflip(label)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ColorJitter(T.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.):
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img = sample['image']
            if len(img.shape) == 4:
                img = img.permute(3, 0, 1, 2).contiguous()
            elif img.size(0) == 1:
                img = img.repeat(3, 1, 1)
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

            if len(img.shape) == 4:
                img = img.permute(1, 2, 3, 0).contiguous()
                img = img[0].unsqueeze(0)
            sample['image'] = img
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
