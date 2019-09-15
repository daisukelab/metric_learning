from dlcliche.fastai import *
import PIL
from PIL import ImageDraw
from PIL import ImageFilter


class AnomalyTwinImageList(ImageList):
    """ImageList that doubles 'true' label images as 'false' twin.
    Artificially generated twin will have a small scar on the image
    to simulate that a defect happened to be there on the image.

    Feed 'true' labeled images only.
    """
    SIZE = 224
    WIDTH_MIN = 1
    WIDTH_MAX = 16
    LENGTH_MAX = 225 // 5
    COLOR = True

    @classmethod
    def set_params(cls, width_min=1, width_max=16, length=225//5, color=True):
        cls.WIDTH_MIN, cls.WIDTH_MAX = width_min, width_max
        cls.LENGTH_MAX, cls.COLOR = length, color

    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)

    def get(self, i):
        image = PIL.Image.open(self.items[i])
        image = image.resize((self.SIZE, self.SIZE))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if (i % 2) != 0: # odd = defect twin
            image = self.anomaly_twin(image)
        return Image(pil2tensor(image, np.float32).div_(255))
    
    def random_pick_point(self, image):
        # Randomly choose a point from entire image
        return random.randint(0, self.SIZE), random.randint(0, self.SIZE)

    def anomaly_twin(self, image):
        """Default anomaly twin maker."""
        scar_max = self.LENGTH_MAX
        half = self.SIZE // 2
        # Randomly choose a point on object
        x, y = self.random_pick_point(image)
        # Randomly choose other parameters
        dx, dy = random.randint(0, scar_max), random.randint(0, scar_max)
        x2, y2 = x + dx if x < half else x - dx, y + dy if y < half else y - dy
        c = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        if not self.COLOR: c = (c[0], c[0], c[0])
        w = random.randint(self.WIDTH_MIN, self.WIDTH_MAX)
        ImageDraw.Draw(image).line((x, y, x2,y2), fill=c, width=w)
        return image

    @classmethod
    def databunch(cls, path, images=None, size=224,
                  tfms=None, valid_pct=0.2, extension='.png', confirm_samples=0):
        """
        Arguments:
            path: Root path to the image files.
            images: Predetermined image path name list,
                or setting None will search all files that matches extension.
            size: Image size.
            tfms: Transforms to augment images.
            valid_pct: Percentage to assign samples to validation set.
            extension: File extension of image files.
            confirm_samples: Number of samples to confirm how samples are assigned.
        """
        path = Path(path)
        # Make list of images if not there
        if images is None:
            images = [str(f).replace(str(path)+'/', '') for f in path.glob('**/*'+extension)]
        # Double the image list
        images = [f for ff in images for f in [ff, ff]]
        # Assign labels, and valid sample index
        N = len(images)//2
        labels = [l for _ in range(N) for l in ['normal', 'anomaly']]
        valid_idx = [i for ii in random.sample(range(N), int(N * valid_pct)) for i in [ii*2, ii*2+1]]
        # Make databunch
        df = pd.DataFrame({'filename': images, 'label': labels})
        if confirm_samples > 0:
            print('Example of sample assignment:')
            display(df[:confirm_samples])
        cls.SIZE = size
        return (cls.from_df(df, path)
                .split_by_idx(valid_idx=valid_idx)
                .label_from_df()
                .transform(tfms=tfms, size=size)
                .databunch(no_check=True))


class DefectOnBlobImageList(AnomalyTwinImageList):
    """Derived from AnomalyTwinImageList class,
    this will draw a scar line on the object blob.

    Effective for images with single object like zoom up photo of a single object
    with single-colored background; Photo of a screw on white background for example.

    Note: Easy algorithm is used to find blob, could catch noises; increase BLOB_TH to avoid that.
    """
    BLOB_TH = 20
    WIDTH_MAX = 14

    @classmethod
    def set_params(cls, blob_th=20, width_min=1, width_max=14, length=225//5, color=True):
        cls.BLOB_TH = blob_th
        cls.WIDTH_MIN, cls.WIDTH_MAX = width_min, width_max
        cls.LENGTH_MAX, cls.COLOR = length, color

    def random_pick_point(self, image):
        # Randomly choose a point on object blob
        np_img = np.array(image.filter(ImageFilter.SMOOTH)).astype(np.float32)
        ys, xs = np.where(np.sum(np.abs(np.diff(np_img, axis=0)), axis=2) > self.BLOB_TH)
        x = random.choice(xs)
        ys_x = ys[np.where(xs == x)[0]]
        y = random.randint(ys_x.min(), ys_x.max())
        return x, y


class DefectOnTheEdgeImageList(DefectOnBlobImageList):
    """Derived from DefectOnBlobImageList class, this simulates
    that object have a defect on the _EDGE_ of it.

    Effective for images with single object like photo of zoom up of a single screw,
    which could have defects on the edge.

    Note: All the edges could be target, including edges inside the object.
    """

    def random_pick_point(self, image):
        # Randomly choose a point on edge (where any color change happens)
        np_img = np.array(image.filter(ImageFilter.SMOOTH)).astype(np.float32)
        ys, xs = np.where(np.sum(np.abs(np.diff(np_img, axis=0)), axis=2) > self.BLOB_TH)
        obj_pts = (ys, xs)
        obj_pts = [(x, y) for y, x in zip(*obj_pts)]
        x, y = (obj_pts[random.randint(0, len(obj_pts) - 1)]
                if len(obj_pts) > 0 else (self.SIZE//2, self.SIZE//2))
        return x, y
