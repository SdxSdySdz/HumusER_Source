import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader, Dataset
from PIL import Image

import numpy as np
import torch.nn as nn

import cv2
import torch
import torchvision
import glob


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(170496, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, xb):
        # outputs = torch.sigmoid(self.network(xb))
        # return outputs
        return self.network(xb)

    def forward_image(self, image_path):
        original = Image.open(image_path)
        resized = original.resize(image_taget_shape, Image.ANTIALIAS)
        img = np.array(resized.convert('L'))

        furier_image = np.fft.fft2(img)
        scectrum = np.log(np.abs(furier_image))
        phases = np.angle(furier_image)

        result = torch.from_numpy(np.vstack((scectrum, phases)))[None, None, :]
        result = self.forward(result.float())
        return result

    def training_step(self, batch):
        # images, labels = batch
        # out = self(images) * 100 * percentage_multiplier # Generate predictions

        images, labels = batch
        out = self(images)

        print(f'OUT: {torch.flatten(out)}')
        print(f'ACT: {labels}')

        loss = torch.nn.functional.mse_loss(out, labels.float())  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = torch.nn.functional.mse_loss(out, labels)  # Calculate loss
        # acc = accuracy(out, labels)           # Calculate accuracy
        # return {'val_loss': loss.detach(), 'val_acc': acc}

        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        # batch_accs = [x['val_acc'] for x in outputs]
        # epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        # return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        # print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        #     epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        print("!!!!!!!!!!!!!!Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))


class SoilsDataset(Dataset):
    def __init__(self, soil_gray_scale_images, humus_percentages, image_shape, transforms=None):
        if len(soil_gray_scale_images) != len(humus_percentages):
            raise Exception()

        self.soil_gray_scale_images = soil_gray_scale_images;
        self.humus_percentages = humus_percentages
        self.image_shape = image_shape
        self.transforms = transforms

    def __len__(self):
        return len(self.humus_percentages)

    def __getitem__(self, i):
        img = cv2.resize(self.soil_gray_scale_images[i], self.image_shape)

        if self.transforms is not None:
            img = self.transforms(img)

        furier_image = np.fft.fft2(img)
        scectrum = np.log(np.abs(furier_image))
        phases = np.angle(furier_image)

        result = torch.from_numpy(np.vstack((scectrum, phases)))[None, :]
        # result = result.permute(2, 0, 1).float()

        return result.float(), float(self.humus_percentages[i])

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return torchvision.transforms.functional.rotate(x, self.angle)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        print(f'===Epoch {epoch}===')
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history

image_taget_shape = (150, 150)
class MainApplication(tk.Frame):
    def __init__(self, is_model_pretrained: bool, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        if is_model_pretrained:
            self._model = Model()  # we do not specify pretrained=True, i.e. do not load default weights
            self._model.load_state_dict(torch.load('model_weights.pth'))
            self._model.eval()
        else:
            tk.messagebox.showinfo(title="Warning", message="Your model is not pretrained.\nWhen you choose your soil image\nand click \"Predict\" button\ntraining will start")
            self._model = None

        self.parent = parent
        self._soil_file_name = None
        self.last_image = None

        self._img_size = (400, 400)
        img = ImageTk.PhotoImage(Image.fromarray(np.ones(self._img_size)))

        self.soil_display = tk.Label(self, image=img, background='#323232')
        self.magnitudes_display = tk.Label(self, image=img, background='#323232')
        self.phases_display = tk.Label(self, image=img, background='#323232')

        self.loading_image_button = tk.Button(
            self,
            text='Load soil',
            width=20,
            command=self._on_loading_image_button_clicked,
            background='#FFFB36'
        )

        self._humus_label = tk.Label(self, text='Humus level:')

        self.prediction_button = tk.Button(
            self,
            text='Predict',
            width=30,
            command=self._on_prediction_button_clicked
        )

        self.soil_display.grid(row=0, column=0, padx=50)
        self.magnitudes_display.grid(row=0, column=1, padx=20)
        self.phases_display.grid(row=0, column=2, padx=20)
        self.loading_image_button.grid(row=1)
        self._humus_label.grid(row=2, column=1)
        self.prediction_button.grid(row=3, column=1)

    @property
    def _is_model_pretrained(self):
        return self._model is not None

    def _on_loading_image_button_clicked(self):
        self._soil_file_name = filedialog.askopenfilename(
            initialdir="/",
            title="Select A Soil Image",
            filetype=(("jpeg files","*.jpg"),("all files","*.*"))
        )
        if not self._soil_file_name.endswith('.jpg'):
            self._soil_file_name = None
            self.last_image = None
            return

        original = Image.open(self._soil_file_name)
        self.last_image = original

        resized = original.resize(self._img_size, Image.ANTIALIAS)
        image = ImageTk.PhotoImage(resized)

        self.soil_display.configure(image=image)
        self.soil_display.image = image

        img_c1 = np.array(original.convert('L'))
        img_c2 = np.fft.fft2(img_c1)
        img_c3 = np.fft.fftshift(img_c2)

        spectrum = np.log(1+np.abs(img_c3))
        phases = np.angle(img_c2)

        spectrum = (spectrum / np.max(spectrum)) * 255

        original = Image.fromarray(spectrum)
        resized = original.resize(self._img_size, Image.ANTIALIAS)
        image = ImageTk.PhotoImage(resized)

        self.magnitudes_display.configure(image=image)
        self.magnitudes_display.image = image

        phases = ((phases - np.min(phases)) / (np.max(phases) - np.min(phases))) * 255

        original = Image.fromarray(phases)
        resized = original.resize(self._img_size, Image.ANTIALIAS)
        image = ImageTk.PhotoImage(resized)

        self.phases_display.configure(image=image)
        self.phases_display.image = image

    def _on_prediction_button_clicked(self):
        if self._soil_file_name is None:
            return

        image = np.array(self.last_image)
        if not self._is_model_pretrained:
            self._model = self._get_trained_model()

        percentage = float(self._model.forward_image(self._soil_file_name)[0][0])
        percentage_string = str(percentage)[:5]
        self._humus_label.configure(text=f'Humus level: {percentage_string}%')

    def _get_trained_model(self):
        image_paths = glob.glob('Soils/*.jpg')
        if len(image_paths) == 0:
            raise Exception()

        image_paths = sorted(image_paths, key=lambda path: int(path.replace('Soils\\', '').replace('Soils/', '').replace('.jpg', '')))
        soil_gray_scale_images = [cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE) for i in range(len(image_paths))]

        humus_percentages = [
            1.65,
            1.77,
            1.55,
            1.8,
            1.9,
            2.2,
            1.55,
            2.31,
            1.95,
            2.3,
            1.2,
            2.66,
            3.3,
            2.5,
            1.55,
            1.55,
            1.5,
            1.5,
            1.72,
            1.65,
            1.65,
            1.4,
        ]

        humus_percentages = (torch.tensor(humus_percentages)).float()

        if len(humus_percentages) != len(soil_gray_scale_images):
            raise Exception()

        # original
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(image_taget_shape),
            torchvision.transforms.ToTensor(),
        ])
        original_dataset = SoilsDataset(soil_gray_scale_images, humus_percentages, image_taget_shape)

        # rotations


        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            RotationTransform(90),
        ])
        rotation90_dataset = SoilsDataset(soil_gray_scale_images, humus_percentages, image_taget_shape, transforms)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            RotationTransform(180),
        ])
        rotation180_dataset = SoilsDataset(soil_gray_scale_images, humus_percentages, image_taget_shape, transforms)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            RotationTransform(270),
        ])
        rotation270_dataset = SoilsDataset(soil_gray_scale_images, humus_percentages, image_taget_shape, transforms)

        # union
        datasets = [
            original_dataset,
            rotation90_dataset,
            rotation180_dataset,
            rotation270_dataset,

            # cropping_dataset,
            # flipping_dataset,
            # brightness_dataset,
            # noise_dataset,
            # erasing_dataset,
        ]

        dataset = torch.utils.data.ConcatDataset(datasets)
        dataloader = DataLoader(dataset=datasets, shuffle=True)

        print(len(dataset))

        train_percent = 0.8
        train_size = int(train_percent * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

        batch_size = 32

        loaders = {
            'train': torch.utils.data.DataLoader(train_data,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=1),

            'test': torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=1),
        }

        model = Model()
        num_epochs = 50
        opt_func = torch.optim.Adam
        lr = 0.001
        # fitting the model on training data and record the result after each epoch
        history = fit(num_epochs, lr, model, loaders['train'], loaders['test'], opt_func)
        torch.save(model.state_dict(), 'model_weights.pth')

        return self._model
