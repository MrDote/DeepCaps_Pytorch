from torch.utils.data import Dataset, DataLoader
import numpy as np
# import pandas as pd
# import os
# from skimage import io
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import cfg
# import torch


class SDSSData(Dataset):
    def __init__(self, data_path, train, labels_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """ 
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.img_shape = cfg.IMG_SHAPE

        self.labels = np.load(labels_file)
        # self.labels = pd.read_csv(labels_file)


        #* for image folder
        # self.img_paths = self.labels['GalaxyID'].apply(lambda row : self.data_path + row)
        # self.data = [io.imread(img_path) for img_path in self.img_paths]

        #* for npy file
        self.data = np.load(data_path)




        self.data = self.data[:100]
        self.labels = self.labels[:100]

        # print(self.labels.shape)
        # print(self.data.shape)




        data_train, data_test, labels_train, labels_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        if train:
            self.data = data_train
            self.labels = labels_train
        else:
            self.data = data_test
            self.labels = labels_test

        self.data = np.vstack(self.data).reshape(-1, cfg.COLORS, self.img_shape, self.img_shape)
        self.data = self.data.transpose((0, 2, 3, 1))
        # print(self.data.shape)
        

        # self.labels = self.labels['ng'].tolist()
        self.labels = [int(i) for i in self.labels]



    #This will return a given image and a corrosponding index for the image
    #__getitem__ to support the indexing such that dataset[i] can be used to get ith sample.
    def __getitem__(self, index):
        # img = io.imread(self.data[index])
        # print(self.data[index].shape)
        # img = Image.fromarray(np.squeeze(self.data[index], axis=2))
        img = Image.fromarray(self.data[index], mode="RGB")

        labels = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, labels


    #len(dataset) returns the size of the dataset
    #The __len__ function returns the number of samples in our dataset
    def __len__(self):
        return len(self.labels) #number of images/Entries in csv file



class SDSS:

    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rotation_degrees=45, translate=(0.1, 0.1), scale=(0.95, 1.1)):

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.num_workers = num_workers

        self.img_size = cfg.IMG_SHAPE
        self.num_class = 2
        self.labels = cfg.LABELS_FILE

    def __call__(self):

        train_loader = DataLoader(SDSSData(
            data_path=self.data_path,
            train=True,
            labels_file=self.labels,
            
            transform=transforms.Compose([
                transforms.RandomAffine(
                    degrees=self.rotation, 
                    translate=self.translate,
                    scale=self.scale
                ),
                transforms.ToTensor(),
                transforms.Resize(self.img_size)
            ]),
            ),
            drop_last=True,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

        test_loader = DataLoader(SDSSData(
                data_path=self.data_path,
                train=False,
                labels_file=self.labels,
                transform=transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                ])
            ),

            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True
        )

        return train_loader, test_loader, self.img_size, self.num_class

# train_loader, test_loader, img_size, num_class = SDSS(data_path=cfg.DATASET_FOLDER, batch_size=cfg.BATCH_SIZE, shuffle=False)()

# if __name__ == '__main__':
#     for batch_idx, (train_data, labels) in enumerate(train_loader): #from training dataset
#         data, labels = train_data[0], labels
#         print(data)


########  VIEW ORIGINAL VS TRANSFORMED  #############
# original = transformed_dataset[0]['labels']
# imgs = [transforms.Grayscale(num_output_channels=1)(original)]
# imgs.insert(0, original)
# imgs = [toPIL(img) for img in imgs]

# fig = plt.figure()
# n = len(imgs)

# for i in range(n):
#     fig.add_subplot(1, n, i+1)
#     plt.imshow(imgs[i])
# plt.show()





# import matplotlib.pyplot as plt
# samples = 6
# starting_index = 35

# fig, axs = plt.subplots(2, samples, figsize=(10,7))
# fig.subplots_adjust(wspace=0.1, hspace=0.0)
# axs = axs.ravel()

# toPIL = transforms.ToPILImage()

# for i in range(starting_index, starting_index + samples):
#     print(train_loader.dataset[i][1])
#     original = train_loader.dataset[i][0]
#     transformed = transforms.RandomAffine(
#         degrees=15,
#         translate=(0.15,0.15),
#         scale=(0.95, 1.1)
#     )(original)
    
#     ii = (i-starting_index)

#     axs[ii].imshow(toPIL(original))
#     axs[ii].axis('off')

#     axs[ii+samples].imshow(toPIL(transformed))
#     axs[ii+samples].axis('off')

# plt.show()



# if __name__ == '__main__':
#     print(len(train_loader.dataset))
#     print(len(test_loader.dataset))