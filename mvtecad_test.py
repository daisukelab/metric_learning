from utils import *
from anomaly_twin_imagelist import *
from metric_nets import *


def subcase_MVTecAD(testcases, subs, case_no, sub_no):
    return f'case{case_no}-{sub_no}-{testcases[case_no]}-{subs[case_no][sub_no]}'

def clean_all_subcase_MVTecAD(path):
    """Clean up all test data files/folders."""
    for d in path.glob('case*'):
        if not d.is_dir(): continue
        ensure_delete(d)

def create_subtests_MVTecAD(path, subtest_type, testcases, skip_data_creation=False):
    """
    Creates sub test case folders.
    """
    def _make_case_folder(case_no, sub_no):
        subcase = subcase_MVTecAD(testcases, subs, case_no, sub_no)
        case_folder = path/subcase
        # Prepare folders
        ensure_delete(case_folder)
        ensure_folder(case_folder)
        ensure_folder(case_folder/'train')
        ensure_folder(case_folder/'test')
        return subcase, case_folder
    def _print_content(subcase, case_folder):
        print(f'# Test: {subcase}')
        print([d.parent.name+'/'+d.name for d in (case_folder/'train').glob('*')])
        print([d.parent.name+'/'+d.name for d in (case_folder/'test').glob('*')])
        print()

    # TEST TYPE 1: Use artificially generated samples (from train samples) as anomaly class.
    # TEST TYPE 2: Use one of test anomaly class to train with good, test others.
    if subtest_type == 'artificial' or subtest_type == 'out_of_folds':
        # Make all test combinations
        subs = []
        for tc in testcases:
            cur = sorted([d.name for d in (path/f'original/{tc}/test').glob('[A-Za-z0-9]*')
                          if d.name != 'good'])
            subs.append(cur)
        # Build test folders
        if not skip_data_creation:
            for case_no in range(len(testcases)):
                for sub_no in range(len(subs[case_no])):
                    case = testcases[case_no]
                    sub = subs[case_no][sub_no]
                    subcase, case_folder = _make_case_folder(case_no, sub_no)
                    # Copy samples
                    copy_any(path/f'original/{case}/train/good', case_folder/'train')
                    copy_any(path/f'original/{case}/test/{sub}', case_folder/'train')
                    copy_any(path/f'original/{case}/test/good', case_folder/'test')
                    for test_sub in subs[case_no]:
                        copy_any(path/f'original/{case}/test/{test_sub}', case_folder/'test')
                    _print_content(subcase, case_folder)
    # TEST TYPE 3: Use some test cases' good class to train.
    elif subtest_type == 'simple_mix':
        testcases = ['capsule', 'carpet', 'leather', 'cable']
        subs = [['simple_mix'], ['simple_mix'], ['simple_mix'], ['simple_mix']]
        for case_no in range(len(testcases)):
            subcase, case_folder = _make_case_folder(case_no=case_no, sub_no=0)
            for case in testcases:
                copy_any(path/f'original/{case}/train/good', case_folder/f'train/{case}')
            sub = testcases[case_no]
            copy_any(path/f'original/{sub}/test', case_folder/f'test/{sub}')
            _print_content(subcase, case_folder)
    # UNKNOWN TYPE
    else:
        raise Exception(f'Unknown test type: {subtest_type}')

    return testcases, subs

def databunch_MVTeczAD(path, subtest_type, testcases, subs, case_no, train_valid_pct, img_size,
                       artificial_image_list_cls=AnomalyTwinImageList, tfms=None):
    """
    Creates sub test case databunch.
    """
    subcase = subcase_MVTecAD(testcases, subs, case_no, 0) # TODO: Implement for sub_no in the future...

    if tfms is None:
        # Transforms:
        # - No reflection in crop_pad, harmful for textures.
        # - No flip
        tfms_trn, tfms_val = get_transforms(do_flip=False)
        tfms = (tfms_trn[:-1].append(crop_pad(padding_mode='zeros')), tfms_val)

    # TEST TYPE 1: Use artificially generated samples (from train samples) as anomaly class.
    if subtest_type == 'artificial':
        data = artificial_image_list_cls.databunch(path/f'{subcase}/train/good',
                                                   size=img_size, tfms=tfms)
    # TEST TYPE 2: Use one of test anomaly class to train with good, test others.
    # TEST TYPE 3: Use some test cases' good class to train.
    elif subtest_type == 'out_of_folds' or subtest_type == 'simple_mix':
        # Create sample list.
        files  = [Path(f) for f in (path/f'{subcase}/train').glob('**/[A-Za-z0-9]*.png')]
        labels = [Path(f).parent.name for f in files]
        assert len(files) > 0, f"{path/f'{subcase}/train'} has no images."
        # Balance samples between classes.
        balanced_files, balanced_labels = balance_class_by_over_sampling(files, labels)
        # Then databunch from the balanced training samples.
        data = ImageDataBunch.from_lists(path/subcase, balanced_files, balanced_labels,
                                         valid_pct=train_valid_pct,
                                         test='test', ds_tfms=tfms, size=img_size)
    # UNKNOWN TYPE
    else:
        raise Exception(f'Unknown test type: {subtest_type}')
    return data

from utils import *
from anomaly_twin_imagelist import *
from metric_nets import *


class MVTecADTest:
    "Anomaly detection test class using MVTec AD dataset"

    def __init__(self, path, test_type='artificial', distance='cosine', n_mosts=5,
                 train_valid_pct=0.2, test_size=1.0, img_size=224, pred_fn=np.min, skip_data_creation=False,
                 artificial_image_list_cls=AnomalyTwinImageList,
                 testcases=None):
        """
        Args:
            base_databunch: Databunch fast.ai class object that holds whole dataset.
            path
            test_type
            distance: 'cosine' or 'euclidean'
            n_mosts: Number of samples to show worst cases.
            subsample_size: (0, 1) or 1 or integer to set size of subsampling train/valid sets.
            train_valid_pct
            test_size: (0, 1) or 1 or integer to set size of subsampling test set.
            pred_fn: Function to predict distance; np.min() by default.
        """
        self.path, given_testcases = prepare_MVTecAD(path)
        self.testcases = given_testcases if testcases is None else testcases
        self.test_type, self.cur_test = test_type, None
        self.distance, self.n_mosts, self.train_valid_pct = distance, n_mosts, train_valid_pct
        self.test_size, self.img_size, self.pred_fn = test_size, img_size, pred_fn
        self.artificial_image_list_cls = artificial_image_list_cls
        self.results = {}
        self.create_test_data(skip_data_creation)

    @property
    def n_cases(self): return len(self.testcases)

    def case_folder(self, case_no, sub_no):
        return self.path/subcase_MVTecAD(self.testcases, self.subs, case_no, sub_no)

    def set_test(self, case_no, sub_no):
        self.cur_test = [case_no, sub_no]
        self.test_path = self.case_folder(case_no, sub_no)
        
    def no_test_set_yet(self):
        result = self.cur_test is None
        if result: print('ERROR: Call set_test first!')

    def databunch(self):
        if self.no_test_set_yet(): return None
        return databunch_MVTeczAD(self.path, subtest_type=self.test_type, testcases=self.testcases,
                                  subs=self.subs, case_no=self.cur_test[0],
                                  train_valid_pct=self.train_valid_pct, img_size=self.img_size,
                                  artificial_image_list_cls=self.artificial_image_list_cls)

    def create_test_data(self, skip_data_creation=False):
        self.testcases, self.subs = create_subtests_MVTecAD(self.path,
                                                            subtest_type=self.test_type,
                                                            testcases=self.testcases,
                                                            skip_data_creation=skip_data_creation)

    def test_title(self):
        if self.no_test_set_yet(): return '(no test)'
        # TEST TYPE 1: Use artificially generated samples (from train samples) as anomaly class.
        # TEST TYPE 2: Use one of test anomaly class to train with good, test others.
        if self.test_type == 'artificial' or self.test_type == 'out_of_folds':
            return self.testcases[self.cur_test[0]]
        # TEST TYPE 3: Use some test cases' good class to train.
        elif self.test_type == 'simple_mix':
            return self.subs[0][self.cur_test[1]]
        else:
            raise Exception(f'Unknown test type: {self.test_type}')

    def eval_ds_dl(self, sub_folder):
        return prepare_subset_ds_dl(self.test_path/sub_folder, size=self.test_size,
                                    tfms=None, img_size=self.img_size)

    def store_results(self, name, result, case_no, sub_no):
        if name not in self.results:
            self.results[name] = [[None for _ in range(len(self.subs[case_no]))]
                                  for cn in range(self.n_cases)]
        self.results[name][case_no][sub_no] = result

    def test(self, name, learner_fn, visualize_now=True, vis_class=-1):
        # Train learner
        anomaly_data = self.databunch()
        learn = learner_fn(anomaly_data)

        print('Calculating embeddings for all test and train samples.')
        eval_test_ds, eval_test_dl = self.eval_ds_dl('test')
        eval_train_ds, eval_train_dl = self.eval_ds_dl(learn.data.train_ds.path)
        test_embs,  test_y  = get_embeddings(body_feature_model(learn.model), eval_test_dl, return_y=True)
        train_embs, train_y = get_embeddings(body_feature_model(learn.model), eval_train_dl, return_y=True)

        distances = n_by_m_distances(test_embs, train_embs, how=self.distance)
        print(f'Calculated distances in shape (test x train) = {distances.shape}')

        # Get basic values
        test_anomaly_mask = [y != eval_test_ds.classes.index('good') for y in test_y]
        test_anomaly_idx = np.where(test_anomaly_mask)[0]
        y_true = np.array(list(map(int, test_anomaly_mask)))
        preds = self.pred_fn(distances, axis=1)

        # Get worst/best info
        def get_test_xx_most_info(most_test_idxs):
            most_train_idxs = np.argmin(distances[most_test_idxs], axis=1)

            most_train_info = eval_train_ds.to_df().iloc[most_train_idxs]
            most_test_info  = eval_test_ds.to_df().iloc[most_test_idxs]
            #print(distances.shape, most_train_idxs, most_test_idxs)
            most_test_info['distance'] = [distances[_test, _trn]
                                           for _trn, _test in zip(most_train_idxs, most_test_idxs)]
            most_test_info['train_idx'] = most_train_info.index
            most_test_info['train_x'] = most_train_info.x.values
            most_test_info['train_y'] = most_train_info.y.values
            return most_test_info

        # 1. Get worst case
        preds_y1 = preds[test_anomaly_mask]
        worst_test_idxs = test_anomaly_idx[preds_y1.argsort()[:self.n_mosts]]
        worst_test_info = get_test_xx_most_info(worst_test_idxs)

        # 2. ROC/AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_true, preds)
        auc = metrics.auc(fpr, tpr)

        # 3. Get mean_class_distance
        mean_class_distance = [[np.mean(distances[test_y == cur_test_y, :])]
                               for cur_test_y in range(eval_test_dl.c)]
        distance_df = pd.DataFrame(mean_class_distance, columns=[self.test_title()])
        distance_df.index = eval_test_ds.classes

        result = distance_df, (auc, fpr, tpr), worst_test_info
        case_no, sub_no = self.cur_test
        self.store_results(name, result, case_no, sub_no)
        
        # Visualize
        if visualize_now:
            # Results
            display(distance_df)
            print(f'AUC = {auc}')
            # Embeddings
            visualize_embeddings(title='Class embeddings distribution', embeddings=test_embs,
                                 ys=test_y, classes=eval_test_ds.y.classes)
            plt.show()
            # Best/Worst cases per class
            if vis_class == -1:
                vis_class = 1 if eval_test_ds.classes[0] == 'good' else 0
            for cls in range(eval_test_ds.c):
                if vis_class is not None: # None = all
                    if eval_test_ds.classes[cls] == 'good': continue
                    if vis_class != cls: continue
                test_mask = [y == cls for y in test_y]
                test_idx = np.where(test_mask)[0]
                preds_y1 = preds[test_mask]

                class_worst_test_idxs = test_idx[preds_y1.argsort()[:self.n_mosts]]
                worst_test_info = get_test_xx_most_info(class_worst_test_idxs)
                class_best_test_idxs  = test_idx[preds_y1.argsort()[::-1][:self.n_mosts]]
                best_test_info  = get_test_xx_most_info(class_best_test_idxs)
                if eval_test_ds.classes[cls] == 'good':
                    worst_test_info, best_test_info = best_test_info, worst_test_info

                self.show_test_matching_images('Best: ' + eval_test_ds.classes[cls], learn, best_test_info)
                plt.show()
                self.show_test_matching_images('Worst: ' + eval_test_ds.classes[cls], learn, worst_test_info)
                plt.show()
        
        return result

    def do_tests(self, name, learner_fn, visualize=[0], delete_models=False):
        for case_no in range(self.n_cases):
            visualize_now = (visualize == 'all') or (case_no in visualize)
            print(f'\nTesting {name} for case #{case_no}')
            self.set_test(case_no, 0)
            self.test(name, learner_fn, visualize_now=visualize_now)
        if delete_models:
            delete_saved_models(self.path)

    def show_test_matching_images(self, title, learn, most_test_info, case_no=None, sub_no=None):
        fig, all_axes = plt.subplots(2, 5, figsize=(18, 8), gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(title)
        data_path = self.test_path if case_no is None else self.case_folder(case_no, sub_no)
        for j, axes in enumerate(all_axes):
            for i, ax in enumerate(axes):
                cur = most_test_info.loc[most_test_info.index[i]]
                if j == 0:
                    visualize_cnn_by_cam(learn, ax=ax, label=f'test/{cur.x}\nhas distance={cur.distance:.6f}',
                                         x=pil2tensor(load_rgb_image(data_path/f'test/{cur.x}')/255,
                                                      np.float32).cuda(), y=0)
                else:
                    show_np_image(load_rgb_image(learn.data.train_ds.path/cur.train_x), ax=ax)
                    ax.set_title(f'from good {cur.train_x}')

    def test_summary(self, results=None, names=None, auc_range=[0.2,1.0], dist_range=None):
        if results is None:
            names = self.results.keys()
            results = list(self.results.values())
        normal_dists = {name: [] for name in names}
        anomaly_dists = {name: [] for name in names}
        aucs = pd.DataFrame()
        for result, name in zip(results, names):
            for case_no in range(self.n_cases):
                distance_df, (auc, fpr, tpr), worst_test_info = result[case_no][0]
                # collect distances
                anomaly_dists[name].extend(distance_df[distance_df.index != 'good'].values[:, 0])
                normal_dists[name].extend(distance_df[distance_df.index == 'good'].values[:, 0])
                # collect auc
                aucs.loc[case_no, name] = auc
        aucs.index = self.testcases
        # distance metric
        distance_norms = pd.DataFrame(normal_dists).mean()
        normalized_anomaly_distances = pd.DataFrame(anomaly_dists)/distance_norms
        normalized_normal_distances = pd.DataFrame(normal_dists)/distance_norms

        print('# Stat: AUC')
        ax = aucs.boxplot(figsize=(5, 4), rot=30)
        if auc_range is not None: ax.set_ylim(*auc_range)
        plt.show()

        display(df_apply_sns_color_map(aucs, reverse=True))
        plt.show()

        print('# Stat: Normalized distances')
        ax = barplot_paired_charts(normalized_anomaly_distances,
                                   normalized_normal_distances, 'Anomaly', 'Normal',
                                   figsize=(10, 5))
        if dist_range is not None: ax.set_ylim(dist_range)
        plt.show()

        self.normalized_anomaly_distances, self.aucs = normalized_anomaly_distances, aucs
        self.normalized_normal_distances = normalized_normal_distances
        return normalized_anomaly_distances, normalized_normal_distances, aucs
