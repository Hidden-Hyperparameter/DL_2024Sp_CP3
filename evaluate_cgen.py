from scipy.linalg import sqrtm
from tqdm.autonotebook import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import argparse
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gan', action='store_true')
parser.add_argument('--vae', action='store_true')
parser.add_argument('--repeat', type=int, default=1)
args = parser.parse_args()

mnist = torchvision.datasets.MNIST(download=False, train=True, root="./data")

data_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((mnist.data.float().mean() / 255, ),
                         (mnist.data.float().std() / 255, ))
])
if args.gan:
    dataset = ImageFolder('gan/generated', transform=data_transform)
elif args.vae:
    dataset = ImageFolder('vae/generated', transform=data_transform)
else:
    raise NotImplementedError
mnist = torchvision.datasets.MNIST(download=False,
                                   train=True,
                                   root="./data",
                                   transform=data_transform)
mnist_indices = []
for number in range(10):
    mnist_indices.append([
        i for i, is_number in enumerate(mnist.targets == number) if is_number
    ])


class MnistInceptionV3(nn.Module):

    def __init__(self, in_channels=3):
        super(MnistInceptionV3, self).__init__()

        self.model = models.inception_v3(pretrained=True)

        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) -
            2 * tr_covmean)


# prepare model
model = MnistInceptionV3()
model.load_state_dict(torch.load("MnistInceptionV3.pth"))

model.model.fc = nn.Identity()
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
model = model.to(device)

fids = np.zeros((args.repeat, 10), dtype=np.float32)

for repeat in range(args.repeat):
    with torch.no_grad():
        for number in tqdm(range(10)):
            # from image folder
            generated_sampler = torch.utils.data.SubsetRandomSampler(
                list(range(100 * number, 100 * (number + 1))))
            generated_dataloader = torch.utils.data.DataLoader(
                dataset,
                sampler=generated_sampler,
                batch_size=100,
                shuffle=False,
                pin_memory=True)
            generated_img = None
            for img, label in generated_dataloader:
                assert (label == number).all(), (label, number)
                generated_img = img.to(device)
                break

            # from mnist
            mnist_sampler = torch.utils.data.SubsetRandomSampler(
                mnist_indices[number])
            mnist_dataloader = torch.utils.data.DataLoader(
                mnist,
                sampler=mnist_sampler,
                batch_size=100,
                shuffle=False,
                pin_memory=True)
            mnist_img = None
            for img, label in mnist_dataloader:
                assert (label == number).all(), (label, number)
                mnist_img = img.to(device)
                break

            # calculate activations
            act1 = model(mnist_img).cpu().numpy()
            act2 = model(generated_img).cpu().numpy()
            # calculate mean and covariance statistics
            mu1, sigma1 = act1.mean(0), np.cov(act1, rowvar=False)
            mu2, sigma2 = act2.mean(0), np.cov(act2, rowvar=False)
            fid = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

            fids[repeat, number] = fid

print(f"FID score for 10 classes: {fids.mean(0)}")
