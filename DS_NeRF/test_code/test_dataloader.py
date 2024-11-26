import torch
import os
import imageio
import numpy as np
def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
def imread(f):
        if f.endswith('png'):
            # return imageio.imread(f, ignoregamma=True)
            return imageio.imread(f)
        else:
            return imageio.imread(f)
class imageDataset(torch.utils.data.Dataset):
    def __init__(self, basedir,origin,device):
        """
        Itialize the paired dataset object for loading and transforming paired data samples
        from specified dataset folders.

        This constructor sets up the paths to input and output folders based on the specified 'split',
        loads the captions (or prompts) for the input images, and prepares the transformations and
        tokenizer to be applied on the data.

        Parameters:
        - dataset_folder (str): The root folder containing the dataset, expected to include
                                sub-folders for different splits (e.g., 'train_A', 'train_B').
        - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
                       sub-folders and caption files within the dataset folder.
        - image_prep (str): The image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """

        super().__init__()
        img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png')][0]
        sh = imageio.imread(img0).shape
        sfx = ''
        factor = 4
        if factor is not None:
            sfx = '_{}'.format(factor)
            _minify(basedir, factors=[factor])
            factor = factor
        elif height is not None:
            factor = sh[0] / float(height)
            width = int(sh[1] / factor)
            _minify(basedir, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        elif width is not None:
            factor = sh[1] / float(width)
            height = int(sh[0] / factor)
            _minify(basedir, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        else:
            factor = 1

        if not origin: 
            imgdir = os.path.join(basedir, 'images' + sfx)
        else:
            imgdir = os.path.join(basedir, 'images' + sfx + '/RGB_inpainted')

        # mskdir = os.path.join(basedir, 'images' + sfx + '/label')
        # depthdir = os.path.join(basedir, 'images' + sfx + '/Depth_inpainted')

        if not os.path.exists(imgdir):
            print(imgdir, 'does not exist, returning')
            return

        
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                    f.endswith('JPG') or f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png')]
        
        imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
        self.imgs = np.stack(imgs, 0)
        self.device=device
        

        # self.input_folder = os.path.join(dataset_folder)
        # if split == "train":
        #     self.input_folder = os.path.join(dataset_folder, "train_A")
        #     self.output_folder = os.path.join(dataset_folder, "train_B")
        #     captions = os.path.join(dataset_folder, "train_prompts.json")
        # elif split == "test":
        #     self.input_folder = os.path.join(dataset_folder, "test_A")
        #     self.output_folder = os.path.join(dataset_folder, "test_B")
        #     captions = os.path.join(dataset_folder, "test_prompts.json")
        # with open(captions, "r") as f:
        #     self.captions = json.load(f)
        # self.img_names = list(self.captions.keys())
        # self.T = build_transform(image_prep)
        # self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        """
        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "values": a tensor of the input image with pixel values 
            scaled to [-1, 1].

            
        """

        image = self.imgs[idx]
        image = torch.tensor(image).to(self.device)
        # image scale to -1,1
        image = image * 2 - 1
        return {
            "pixel_values": image,
            
        }
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='./data/1',
                        help='where to store ckpts and logs')
    parser.add_argument("--origin", action='store_true',default=True,
                        help="use the original MVIP-nerf, use the inpainted images")
    return parser
def train():
    parser = config_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset_train = imageDataset(basedir=args.basedir,origin=args.origin,device=device)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0,generator=torch.Generator(device=device))
    for i, batch in enumerate(dl_train):
        print(batch["pixel_values"])

if __name__ == '__main__':
   
    torch.set_default_device('cuda')
    train()