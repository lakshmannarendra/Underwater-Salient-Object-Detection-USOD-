# Function to load image, depth, label, and contour paths from a dataset list
def load_list(dataset_list, data_root):
    ...
    # Split the dataset list by '+'
    dataset_list = dataset_list.split('+')
    ...
    return images, depths, labels, contours

# Function to load test images and depths from a specified path
def load_test_list(test_path, data_root):
    ...
    # Check if the test path is 'USOD10K' to set the depth root accordingly
    if test_path in ['USOD10K']:
        ...
    return images, depths

# Custom dataset class for loading images and their corresponding depths, labels, and contours
class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, depth_transform, mode, img_size=None, scale_size=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None):
        ...
        # Load image and depth paths based on the mode (train or test)
        if mode == 'train':
            ...
        else:
            ...
        # Store transformations and other parameters
        self.transform = transform
        ...

    # Method to get a single item from the dataset
    def __getitem__(self, item):
        ...
        # Load image and depth
        image = Image.open(self.image_path[item]).convert('RGB')
        ...
        # If in training mode, apply transformations and augmentations
        if self.mode == 'train':
            ...
            return new_img, new_depth, label_224, label_14, label_28, label_56, label_112, \
                   contour_224, contour_14, contour_28, contour_56, contour_112
        else:
            ...
            return image, depth, image_w, image_h, self.image_path[item]

    # Method to return the length of the dataset
    def __len__(self):
        return len(self.image_path)

# Function to create a data loader for the dataset
def get_loader(dataset_list, data_root, img_size, mode='train'):
    ...
    # Set up transformations based on the mode (train or test)
    if mode == 'train':
        ...
    else:
        ...
    # Create and return the dataset
    if mode == 'train':
        dataset = ImageData(...)
    else:
        dataset = ImageData(...)
    return dataset