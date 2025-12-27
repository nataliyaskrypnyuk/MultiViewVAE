from torch.utils.data import Dataset
import os
from PIL import Image

class CaseImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.case_ids = []

        # Iterate through case folders
        for case_name in sorted(os.listdir(root_dir)):
            case_folder = os.path.join(os.path.join(root_dir, case_name), 'DataFiles')
            if os.path.isdir(case_folder):
                for img_name in os.listdir(case_folder):
                    if img_name.lower().endswith('256.png'):
                        img_path = os.path.join(case_folder, img_name)
                        self.image_paths.append(img_path)
                        self.case_ids.append(case_name + "/" + img_name.split("_")[0] + "/" + img_name)  # Keep case name as ID

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        case_id = self.case_ids[idx]

        if self.transform:
            image = self.transform(image)

        return image, case_id

class CaseMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.case_ids = []
        self.image_paths = []

        # Iterate through case folders
        for case_name in sorted(os.listdir(root_dir)):
            case_folder = os.path.join(os.path.join(root_dir, case_name), 'DataFiles')
            if os.path.isdir(case_folder):
                matches_lower = [f for f in os.listdir(case_folder) if ('lower' in f) and ('256.png' in f)]
                matches_upper = [f for f in os.listdir(case_folder) if ('upper' in f) and ('256.png' in f)]
                if matches_lower:
                    self.case_ids.append(case_name + "/lower")  # Keep case name + lower/upper as ID
                    self.image_paths.append([case_folder + '/' + match for match in matches_lower])
                if matches_upper:
                    self.case_ids.append(case_name + "/upper")  # Keep case name + lower/upper as ID
                    self.image_paths.append([case_folder + '/' + match for match in matches_upper])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        img_paths = self.image_paths[idx]
        images = []
        for img_path in img_paths:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return images, case_id
