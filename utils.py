from fastai.vision import *
from torchvision import datasets, transforms
from torch import nn
import PIL
from tqdm import tqdm
from sklearn import metrics
from collections import defaultdict
from dlcliche.data import *
from dlcliche.math import *
from util_visualize import *


def body_feature_model(model):
    """
    Returns a model that output flattened features directly from CNN body.
    """
    try:
        body, head = list(model.org_model.children()) # For XXNet defined in this notebook
    except:
        body, head = list(model.children()) # For original pytorch model
    return nn.Sequential(body, head[:-1])


def get_embeddings(embedding_model, data_loader, label_catcher=None, return_y=False):
    """
    Calculate embeddings for all samples in a data_loader.
    
    Args:
        label_catcher: LearnerCallback for keeping last batch labels.
        return_y: Also returns labels, for working with training set.
    """
    embs, ys = [], []
    for X, y in data_loader:
        # For each batch (X, y),
        #   Set labels (y) if label_catcher's there.
        if label_catcher:
            label_catcher.on_batch_begin(X, y, train=False)
        #   Get embeddings for this batch, store in embs.
        with torch.no_grad():
            # Note that model's output is not softmax'ed.
            out = embedding_model(X).cpu().detach().numpy()
            out = out.reshape((len(out), -1))
            embs.append(out)
        ys.append(y)
    # Putting all embeddings in shape (number of samples, length of one sample embeddings)
    embs = np.concatenate(embs) # Maybe in (10000, 10)
    ys   = np.concatenate(ys)
    if return_y:
        return embs, ys
    return embs


def prepare_full_MNIST_databunch(data_path=Path('data_MNIST'), tfms=get_transforms(do_flip=False)):
    """Creates MNIST full set databunch."""
    restructured_path = prepare_full_MNIST(data_path)
    return ImageDataBunch.from_folder(restructured_path, ds_tfms=tfms)


def prepare_CIFAR10_databunch(data_path=Path('data_CIFAR10'), tfms=get_transforms(do_flip=False)):
    """Creates CIFAR10 full set databunch."""
    restructured_path = prepare_CIFAR10(data_path)
    return ImageDataBunch.from_folder(restructured_path, ds_tfms=tfms)


def prepare_subset_ds_dl(data_path, size=0.1, tfms=None):
    # Sub-sample files
    files = [Path(f) for f in subsample_files_in_tree(data_path, '*.png', size=size)]
    labels = [Path(f).parent.name for f in files]
    # Once create data bunch
    tmp_data = ImageDataBunch.from_lists(data_path, files, labels, valid_pct=0, ds_tfms=tfms)
    # Create dataloader again so that it surely set `shuffle=False`
    dl = torch.utils.data.DataLoader(tmp_data.train_ds, batch_size=tmp_data.batch_size, shuffle=False)
    dl = DeviceDataLoader(dl, tmp_data.device)
    return tmp_data.train_ds, dl


class ToyAnomalyDetection:
    "Toy Anomaly detection problem class"

    def __init__(self, base_databunch, n_anomaly_labels=1, n_cases=10, distance='cosine',
                 n_worsts=5, test_size=0.5):
        """
        Prerequisite:
            Create base databunch object in advance. Call prepare_full_MNIST_databunch()
            for example. (DATA ROOT)/images/train and valid folders will be used as data source.

        Args:
            base_databunch: Databunch fast.ai class object that holds whole dataset.
            n_anomaly_labels: Number of anomaly labels
            n_cases: Number of test cases; 1 to c
            distance: 'cosine' or 'euclidean'
        """
        self.base_data, self.n_anomaly_labels = base_databunch, n_anomaly_labels
        self.n_cases, self.distance, self.n_worsts, self.test_size = n_cases, distance, n_worsts, test_size
        self.create_test_data()
        self.results = defaultdict(list)

    @property
    def c(self): return self.base_data.c

    @property
    def classes(self): return self.base_data.train_ds.y.classes

    @property
    def path(self): return self.base_data.path

    def anomaly_classes(self, case_no):
        def rotate(l, n):
            return l[n:] + l[:n]
        return rotate(list(range(self.c)), case_no)[:self.n_anomaly_labels]

    def all_classes(self):
        return list(range(self.c))

    def create_test_data(self):
        """
        Creates test case folders for unknown anomaly class detection problem.
        Each test cases removes `n_anomaly_labels` classes from training set,
        and model will detect removed class as anomaly class.

        Output:
            Data_root/images/case[0-9]/train and valid folders.
        """
        data_path = self.path
        for case_no in range(self.n_cases):
            case_folder = data_path/f'case{case_no}'
            ensure_delete(case_folder)
            ensure_folder(case_folder)
            ensure_folder(case_folder/'train')
            ensure_folder(case_folder/'valid')
            for ci in range(self.c):
                if ci in self.anomaly_classes(case_no): continue
                label = self.classes[ci]
                copy_any((data_path/f'train/{label}').absolute(), case_folder/f'train', symlinks=False)
                copy_any((data_path/f'valid/{label}').absolute(), case_folder/f'valid', symlinks=False)
                # Unfortunately symlink doesn't work for fast.ai library, fails in reading list linked data...
                # symlink_file((images/f'train/{label}').absolute(), case_folder/f'train/{label}')
                # symlink_file((images/f'valid/{label}').absolute(), case_folder/f'valid/{label}')

    def databunch(self, case_no, tfms=get_transforms(do_flip=False)):
        """Creates test case ImageDataBunch."""
        return ImageDataBunch.from_folder(self.path/f'case{case_no}', ds_tfms=tfms)

    def store_results(self, name, result):
        self.results[name].append(result)

    def test(self, name, learner_fn, case_no):
        # train learner
        anomaly_data = self.databunch(case_no)
        learn = learner_fn(anomaly_data)

        # Create evaluation data: `test` is evaluation target, while this `train` is subset of training set
        eval_test_ds, eval_test_dl = prepare_subset_ds_dl(self.path/'valid', size=self.test_size, tfms=None)
        eval_train_ds, eval_train_dl = prepare_subset_ds_dl(self.path/f'case{case_no}/train', size=self.test_size, tfms=None)
        test_embs,  test_y  = get_embeddings(body_feature_model(learn.model), eval_test_dl, return_y=True)
        train_embs, train_y = get_embeddings(body_feature_model(learn.model), eval_train_dl, return_y=True)

        print(f'Evaluation size => test:{test_embs.shape}, train{train_embs.shape}')
        distances = n_by_m_distances(test_embs, train_embs, how=self.distance)
        print(f'Calculated distances in shape {distances.shape}')

        # Get basic values
        false_ys = self.anomaly_classes(case_no)
        test_anomaly_mask = [y in  false_ys for y in test_y]
        test_anomaly_idx = np.where(test_anomaly_mask)[0]
        y_true = np.array(list(map(int, test_anomaly_mask)))
        preds = np.min(distances, axis=1)

        # 1. worst case
        preds_y1 = preds[test_anomaly_mask]
        worst_anidxs = preds_y1.argsort()[:self.n_worsts]
        worst_test_idxs = test_anomaly_idx[worst_anidxs]
        worst_train_idxs = np.argmin(distances[worst_test_idxs], axis=1)

        worst_train_info = eval_train_ds.to_df().iloc[worst_train_idxs]
        worst_test_info  = eval_test_ds.to_df().iloc[worst_test_idxs]
        worst_test_info['distance'] = [distances[test_idx, trn_idx]
                                       for trn_idx, test_idx in zip(worst_train_idxs, worst_test_idxs)]
        worst_test_info['train_idx'] = worst_train_info.index
        worst_test_info['train_x'] = worst_train_info.x.values
        worst_test_info['train_y'] = worst_train_info.y.values

        # 2. ROC/AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_true, preds)
        auc = metrics.auc(fpr, tpr)

        # 3. mean_class_distance
        mean_class_distance = [[np.mean(distances[test_y == cur_test_y, :][:, train_y == cur_trn_y])
                                for cur_trn_y in range(eval_train_dl.c)]
                               for cur_test_y in range(eval_test_dl.c)]
        distance_df = pd.DataFrame(mean_class_distance,
                                   columns=[self.classes[c] for c in range(eval_train_dl.c)])
        
        result = distance_df, (auc, fpr, tpr), worst_test_info
        self.store_results(name, result)
        return result
    
    def do_tests(self, name, learner_fn):
        for case_no in range(self.n_cases):
            self.test(name, learner_fn, case_no)

    def test_summary(self, results=None, names=None):
        if results is None:
            names = self.results.keys()
            results = self.results.values()
        normal_dists = {name: [] for name in names}
        anomaly_dists = {name: [] for name in names}
        aucs = pd.DataFrame()
        for case_no in range(self.n_cases):
            for result, name in zip(results, names):
                distance_df, (auc, fpr, tpr), worst_test_info = result[case_no]
                # collect distances
                min_dists = distance_df.min(axis=1).values
                anocls = self.anomaly_classes(case_no)
                anomaly_dists[name].extend(min_dists[anocls])
                normal_dists[name].extend(min_dists[[c for c in self.all_classes() if c not in anocls]])
                # collect auc
                aucs.loc[case_no, name] = auc
        # distance metric
        distance_norms = pd.DataFrame(normal_dists).mean()
        normalized_anomaly_distances = pd.DataFrame(anomaly_dists)/distance_norms

        print('# Stat: normalized anomaly distance')
        display(normalized_anomaly_distances.describe())

        print('# Stat: auc')
        display(aucs.describe())

        # case detail
        for case_no in range(self.n_cases):
            case_df = pd.DataFrame(np.array([r[case_no][0].min(axis=1) for r in results]).T,
                                   columns=names)
            case_df.index = [f'<unk> {c}' if c in self.anomaly_classes(case_no) else str(c)
                             for c in self.all_classes()]
            print(f'\n## {self.anomaly_classes(case_no)}: normalized mean distance')
            display(case_df / distance_norms)

        self.normalized_anomaly_distances, self.aucs = normalized_anomaly_distances, aucs
        return normalized_anomaly_distances, aucs
    
    def save_results(self, to_folder):
        ensure_folder(to_folder)
        self.normalized_anomaly_distances.to_csv(f'{str(to_folder)}/norm-ano-dist_anomaly={self.n_anomaly_labels}'
                                                 +f'_{self.distance}.csv')
        self.aucs.to_csv(f'{str(to_folder)}/auc_anomaly={self.n_anomaly_labels}_{self.distance}.csv')

    def show_worst_test_images(self, title, worst_test_info, case_no):
        fig, all_axes = plt.subplots(2, 5, figsize=(18, 7))
        fig.suptitle(title)
        for j, axes in enumerate(all_axes):
            for i, ax in enumerate(axes):
                cur = worst_test_info.loc[worst_test_info.index[i]]
                if j == 0:
                    img = load_rgb_image(self.path/f'valid/{cur.x}')
                    ax.set_title(f'Failed test/{cur.x}\ndistance={cur.distance:.6f}')
                else:
                    img = load_rgb_image(self.path/f'train/{cur.train_x}')
                    ax.set_title(f'confused w/ {cur.train_x}')
                show_np_image(img, ax=ax)

    def show_all_worst_test_images(self, case_no):
        for name, result in self.results.items():
            distance_df, (auc, fpr, tpr), worst_test_info = result[case_no]
            self.show_worst_test_images(f'{name} in test case #{case_no}', worst_test_info, case_no)
