from experiments.cnn.params import get_params
from instances.psvrt import psvrt
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PSVRTDataset(torch.utils.data.Dataset):
    'Characterizes a custom dataset for PyTorch'

    def __init__(self, n_images, transform):
        'Initialization'
        self.n_images = n_images
        self.transform = transform

        params = get_params()
        self.raw_input_size = params['raw_input_size']
        self.train_data_init_args = params['train_data_init_args']

        self.psvrt = psvrt(self.raw_input_size, self.n_images)
        self.psvrt.initialize_vars(**self.train_data_init_args)

        self.images, self.labels = self.make_images_and_labels(
            self.psvrt, self.n_images, self.raw_input_size)

    def make_images_and_labels(self, psvrt, n_images, raw_input_size):
        'Create a dataset of n images with labels'

        # Create a dataset of images with their labels
        images, labels, _, _ = psvrt.single_batch()

        # Reshape variables for easier interpretation and programming
        images = images.reshape(n_images, raw_input_size[0], raw_input_size[1])
        labels = labels.reshape(n_images, 2)

        # Get only the second column of 'labels' matrix. This col has the relevant labels
        labels = labels[:,1]

        return images, labels

    def __len__(self):
        'Denotes the total number of samples'
        return self.n

    def __getitem__(self, index):
        'Generates one image'

        # Get an image and its label
        image = self.images[index]
        label = self.labels[index]

        # Convert image into PyTorch Tensor
        image = torch.from_numpy(image)

        # Transform image (resize with bilinear interpolation, and normalize with 0.5 mean & std)
        image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = PSVRTDataset(100, transform)
print(dataset[0])
