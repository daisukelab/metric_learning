from dlcliche.fastai import *
import PIL
from PIL import ImageDraw

class AnomalyTwinImageList(ImageList):
    SIZE = 224
    W = 16
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
    
    def anomaly_twin(self, image):
        """Default anomaly twin maker."""
        scar_max = self.SIZE // 5
        half = self.SIZE // 2
        w = min(self.W, scar_max + 1)
        x, y = random.randint(0, self.SIZE), random.randint(0, self.SIZE)
        dx, dy = random.randint(0, scar_max), random.randint(0, scar_max)
        x2, y2 = x + dx if x < half else x - dx, y + dy if y < half else y - dy
        c = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        ImageDraw.Draw(image).line((x, y, x2,y2), fill=c, width=w)
        return image
    
    @staticmethod
    def databunch(path, size=224, tfms=None, valid_pct=0.2, extension='.png', confirm_samples=0):
        """
        Arguments:
            path: Root path to the image files.
            size: Image size.
            tfms: Transforms to augment images.
            valid_pct: Percentage to assign samples to validation set.
            extension: File extension of image files.
            confirm_samples: Number of samples to confirm how samples are assigned.
        """
        path = Path(path)
        images = [str(f).replace(str(path)+'/', '') for ff in path.glob('**/*'+extension) for f in [ff, ff]]
        N = len(images)//2
        labels = [l for _ in range(N) for l in ['normal', 'anomaly']]
        valid_idx = [i for ii in random.sample(range(N), int(N * valid_pct)) for i in [ii*2, ii*2+1]]
        df = pd.DataFrame({'filename': images, 'label': labels})
        if confirm_samples > 0:
            print('Example of sample assignment:')
            display(df[:confirm_samples])
        AnomalyTwinImageList.SIZE = size
        return (AnomalyTwinImageList.from_df(df, path)
                .split_by_idx(valid_idx=valid_idx)
                .label_from_df()
                .transform(tfms=tfms, size=size)
                .databunch(no_check=True))