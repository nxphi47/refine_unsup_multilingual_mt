
import multiprocessing
import time
from typing import Sequence, Tuple
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from fairseq.utils import move_to_cuda
import threading


def get_faiss_gpu_resources(ngpu, mem=5*1024*1024*1024):
    import faiss
    print(f'Create GPU {ngpu} resources with mem={mem}')
    resources = [faiss.StandardGpuResources() for i in range(ngpu)]
    for r in resources:
        r.setTempMemory(size=mem)
    return resources


def faiss_kmeans_gpus(data, k, ngpu, resources=None, fp16=False, mem=5*1024*1024*1024, niter=20, nredo=1, decode_block_size=2 ** 15):
    import faiss
    dim = data.shape[1]
    clus = faiss.Clustering(dim, k)
    clus.verbose = False
    clus.niter = niter
    clus.nredo = nredo
    clus.decode_block_size = int(decode_block_size)
    # print(f'faiss kmeans: {clus.decode_block_size=}') clus.decode_block_size=32768 (default) (2**15)

    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000

    if resources is None:
        resources = get_faiss_gpu_resources(ngpu, mem)

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(resources[0], dim, flat_config[0])
    else:
        indexes = [
            faiss.GpuIndexFlatL2(resources[i], dim, flat_config[i])
            for i in range(ngpu)
        ]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    clus.train(data, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    return resources, index, centroids.reshape(k, dim)


def faiss_kmeans_search(data, centroids):
    import faiss
    dim = data.shape[-1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(centroids)
    preds = idx.search(data, 1)[1][0, 0]
    return preds


def multi_layer_db_search_single(data, kmeans_database, get_search_tree=False, search_tree=None):
    db_centroids = kmeans_database["db_centroids"]
    start_pair, db_dict = kmeans_database["start_pair"], kmeans_database["db_dict"]
    if len(data.shape) == 1:
        data = data[None, :]
    assert data.shape[0] == 1, f'only support single query: {data.shape=}'
    preds = faiss_kmeans_search(data, db_centroids[start_pair[0]:start_pair[1]])
    sub = db_dict[preds]
    if get_search_tree:
        if search_tree is not None:
            search_tree.append(preds)
        else:
            search_tree = [preds]
    if isinstance(sub, tuple):
        sub_pair, sub_dict = sub
        return multi_layer_db_search_single(
            data, {
                "db_centroids": db_centroids,
                "start_pair": sub_pair,
                "db_dict": sub_dict
            }, get_search_tree, search_tree
        )
    else:
        if get_search_tree:
            return sub, search_tree
        return sub


def multi_layer_db_search(data, kmeans_databse):
    assert len(data.shape) == 2
    preds = [
        multi_layer_db_search_single(v, kmeans_databse)
        for i, v in enumerate(np.split(data, len(data), axis=0))
    ]
    return preds


def same_size_cluster(data, distances, topk, k):
    pred_labels = np.full((data.shape[0],), dtype=topk.dtype, fill_value=-1)
    size_per_cluster = data.shape[0] // k
    pred_arange = np.arange(data.shape[0])
    for i in range(k):
        top_pred = topk[:, i]
        top_distance = distances[:, i]
        for j in range(k):
            is_label_j_unlabeled = (top_pred == j) & (pred_labels == -1)
            is_j_labeled = (pred_labels == j)
            is_label_j_unlabeled_sum = is_label_j_unlabeled.astype(int).sum()
            is_j_labeled_sum = is_j_labeled.astype(int).sum()
            if is_j_labeled_sum >= size_per_cluster:
                continue
            remaining = size_per_cluster - is_j_labeled_sum
            if is_label_j_unlabeled_sum > remaining:
                label_j_indices = pred_arange[is_label_j_unlabeled]
                top_distance_j = top_distance[is_label_j_unlabeled]
                sorted_indices = np.argsort(top_distance_j)[::-1]
                selected_indices = sorted_indices[:remaining]
                label_j_selected_indices = label_j_indices[selected_indices]
                pred_labels[label_j_selected_indices] = j
            else:
                pred_labels[is_label_j_unlabeled] = j
    # for the rest, assight top-1
    left = pred_labels == -1
    pred_labels[left] = topk[:, 0][left]
    return pred_labels


def same_size_cluster_rebalance(data, distances, topk, k, centroids):
    # TODO: rebalance may take logs of shit
    pred_labels = np.full((data.shape[0],), dtype=topk.dtype, fill_value=-1)
    size_per_cluster = data.shape[0] // k
    pred_arange = np.arange(data.shape[0])
    for i in range(k):
        top_pred = topk[:, i]
        top_distance = distances[:, i]
        for j in range(k):
            is_label_j_unlabeled = (top_pred == j) & (pred_labels == -1)
            is_j_labeled = (pred_labels == j)
            is_label_j_unlabeled_sum = is_label_j_unlabeled.astype(int).sum()
            is_j_labeled_sum = is_j_labeled.astype(int).sum()
            if is_j_labeled_sum >= size_per_cluster:
                continue
            remaining = size_per_cluster - is_j_labeled_sum
            if is_label_j_unlabeled_sum > remaining:
                label_j_indices = pred_arange[is_label_j_unlabeled]
                top_distance_j = top_distance[is_label_j_unlabeled]
                sorted_indices = np.argsort(top_distance_j)[::-1]
                selected_indices = sorted_indices[:remaining]
                label_j_selected_indices = label_j_indices[selected_indices]
                pred_labels[label_j_selected_indices] = j
            else:
                pred_labels[is_label_j_unlabeled] = j
    # for the rest, assight top-1
    left = pred_labels == -1
    pred_labels[left] = topk[:, 0][left]
    new_centroids = []
    for i in range(k):
        centroid = data[pred_labels == i].mean(0, keepdims=True)
        new_centroids.append(centroid)
    new_centroids = np.concatenate(new_centroids, 0)
    assert len(new_centroids.shape) == 2
    return pred_labels, new_centroids


def rebalance_cluster_assignments(data_size, distances, topk, k):
    # TODO: rebalance may take lots of time
    # TODO: consider multi-processing....
    pred_labels = np.full((data_size,), dtype=topk.dtype, fill_value=-1)
    size_per_cluster = data_size // k
    pred_arange = np.arange(data_size)
    for i in range(k):
        top_pred = topk[:, i]
        top_distance = distances[:, i]
        for j in range(k):
            is_label_j_unlabeled = (top_pred == j) & (pred_labels == -1)
            is_j_labeled = (pred_labels == j)
            is_label_j_unlabeled_sum = is_label_j_unlabeled.astype(int).sum()
            is_j_labeled_sum = is_j_labeled.astype(int).sum()
            if is_j_labeled_sum >= size_per_cluster:
                continue
            remaining = size_per_cluster - is_j_labeled_sum
            if is_label_j_unlabeled_sum > remaining:
                label_j_indices = pred_arange[is_label_j_unlabeled]
                top_distance_j = top_distance[is_label_j_unlabeled]
                sorted_indices = np.argsort(top_distance_j)[::-1]
                selected_indices = sorted_indices[:remaining]
                label_j_selected_indices = label_j_indices[selected_indices]
                pred_labels[label_j_selected_indices] = j
            else:
                pred_labels[is_label_j_unlabeled] = j
    # for the rest, assight top-1
    left = pred_labels == -1
    pred_labels[left] = topk[:, 0][left]
    return pred_labels


def multi_layer_kmeans_gpus_v2(
    data_indices, 
    data, 
    k, 
    max_samples_cluster, 
    retain_max_samples, 
    ngpu, 
    fp16=False, 
    db_centroids=None, 
    mem=5*1024*1024*1024,
    niter=20,
    nredo=1,
    gpu_resources=None,
    max_depth=50,
    cluster_same_size=False,
    current_db_centroids_size=0
):
    # FIXME: this is taking lots of memory gradually, why?
    db_dict = {}
    db_centroids_empty = db_centroids is None   # root level
    # assert retain_max_samples <= max_samples_cluster
    # kmeans
    gpu_resources, index, centroids = faiss_kmeans_gpus(
        data, k, ngpu, 
        resources=gpu_resources,
        fp16=fp16, mem=mem, niter=niter, nredo=nredo
    )
    if cluster_same_size:
        distances, topk = index.search(data, k)
        del index
        pred_labels, centroids = same_size_cluster_rebalance(data, distances, topk, k, centroids)
        del distances, topk
    else:
        pred_labels = index.search(data, 1)[1][:, 0] # # we want to see 1 nearest neighbors
        del index
    
    # filter data
    d_indices_list = []
    d_data_list = []
    centroid_ids_list = []
    for i in range(len(centroids)):
        _is_label_c = pred_labels == i
        if _is_label_c.astype(int).sum() > 0:
            _d_indices = data_indices[_is_label_c]
            _data = data[_is_label_c]
            d_indices_list.append(_d_indices)
            d_data_list.append(_data)
            centroid_ids_list.append(i)
    
    filtered_centroids = np.copy(centroids[centroid_ids_list])
    del centroids
    if db_centroids is None:
        db_centroids = [filtered_centroids]
    else:
        db_centroids.append(filtered_centroids)
    updated_db_centroids_size = current_db_centroids_size + len(filtered_centroids)
    search_pair = (current_db_centroids_size, updated_db_centroids_size)

    intermediate_centroids_size = updated_db_centroids_size
    for i, (_d_indices, _data) in enumerate(zip(d_indices_list, d_data_list)):
        if len(_data) > max_samples_cluster * 2 and max_depth > 0:
            sub_k = min(k, (len(_data) // max_samples_cluster))
            next_level_database, intermediate_centroids_size = multi_layer_kmeans_gpus_v2(
                _d_indices, _data, sub_k, max_samples_cluster, retain_max_samples, 
                ngpu, fp16, db_centroids, mem, niter, nredo,
                gpu_resources=gpu_resources,
                max_depth=max_depth - 1,
                cluster_same_size=cluster_same_size,
                current_db_centroids_size=intermediate_centroids_size
            )
            db_centroids = next_level_database["db_centroids"]
            sub_pair = next_level_database["start_pair"]
            sub_db_dict = next_level_database['db_dict']
            db_dict[i] = (sub_pair, sub_db_dict)
        else:
            db_dict[i] = _d_indices[np.random.permutation(len(_d_indices))[:retain_max_samples]]

    if max_depth <= 0:
        print(f'Max depth reach: {len(data)=}')
    
    print(f'finish sub branch {max_depth=}, {intermediate_centroids_size=}', end='\r')
    output_dict = {
        "db_centroids": db_centroids,
        "start_pair": search_pair,
        "db_dict": db_dict
    }
    if db_centroids_empty:
        print(f'Finally post processing, concatenate db_centroids')
        output_dict['db_centroids'] = np.concatenate(db_centroids, 0)
        assert len(output_dict['db_centroids']) == intermediate_centroids_size, f"{len(output_dict['db_centroids'])=} != {intermediate_centroids_size=}"
        return output_dict
    else:
        return output_dict, intermediate_centroids_size


def multi_layer_db_search_single_svm(data, kmeans_database, get_search_tree=False, search_tree=None):
    db_centroids = kmeans_database["db_centroids"]
    start_pair, db_dict = kmeans_database["start_pair"], kmeans_database["db_dict"]
    predictor = kmeans_database['predictor']
    if len(data.shape) == 1:
        data = data[None, :]
    assert data.shape[0] == 1, f'only support single query: {data.shape=}'
    # preds = faiss_kmeans_search(data, db_centroids[start_pair[0]:start_pair[1]])
    preds = predictor.predict(data)[0]
    sub = db_dict[preds]
    if get_search_tree:
        if search_tree is not None:
            search_tree.append(preds)
        else:
            search_tree = [preds]
    if isinstance(sub, tuple):
        # sub_pair, sub_dict = sub
        sub_pair, sub_db_dict, sub_predictor = sub
        return multi_layer_db_search_single_svm(
            data, {
                'predictor': sub_predictor,
                "db_centroids": db_centroids,
                "start_pair": sub_pair,
                "db_dict": sub_db_dict
            }, get_search_tree, search_tree
        )
    else:
        if get_search_tree:
            return sub, search_tree
        return sub

def multi_layer_db_search_single_predictor(data, kmeans_database, get_search_tree=False, search_tree=None):
    db_dict = kmeans_database["db_dict"]
    predictor = kmeans_database['predictor']
    if len(data.shape) == 1:
        data = data[None, :]
    assert data.shape[0] == 1, f'only support single query: {data.shape=}'
    # preds = faiss_kmeans_search(data, db_centroids[start_pair[0]:start_pair[1]])
    preds = predictor.predict(data)[0]
    sub = db_dict[preds]
    if get_search_tree:
        if search_tree is not None:
            search_tree.append(preds)
        else:
            search_tree = [preds]
    if isinstance(sub, tuple):
        sub_db_dict, sub_predictor = sub
        return multi_layer_db_search_single_predictor(
            data, {
                'predictor': sub_predictor,
                "db_dict": sub_db_dict
            }, get_search_tree, search_tree
        )
    else:
        if get_search_tree:
            return sub, search_tree
        return sub


def multi_layer_kmeans_gpus_svm(
    data_indices, 
    data, 
    k, 
    max_samples_cluster, 
    retain_max_samples, 
    ngpu, 
    fp16=False, 
    db_centroids=None, 
    mem=5*1024*1024*1024,
    niter=20,
    nredo=1,
    gpu_resources=None,
    max_depth=50,
    cluster_same_size=False,
    current_db_centroids_size=0,
    svm_kernel='rbf',
    svm_c=1,
    svm_sub_sample_size=0
):
    # FIXME: this is taking lots of memory gradually, why?
    from sklearn import svm

    db_dict = {}
    db_centroids_empty = db_centroids is None   # root level
    # assert retain_max_samples <= max_samples_cluster
    # kmeans
    data = data.astype("float32")

    gpu_resources, index, centroids = faiss_kmeans_gpus(
        data, k, ngpu, 
        resources=gpu_resources,
        fp16=fp16, mem=mem, niter=niter, nredo=nredo
    )
    if cluster_same_size:
        distances, topk = index.search(data, k)
        del index
        pred_labels, centroids = same_size_cluster_rebalance(data, distances, topk, k, centroids)
        del distances, topk
    else:
        pred_labels = index.search(data, 1)[1][:, 0] # # we want to see 1 nearest neighbors
        del index

    data = data.astype("float32")
    print(f'Start svm training')
    predictor = svm.SVC(kernel=svm_kernel, C=svm_c)
    if svm_sub_sample_size > 0:
        _idx = np.random.permutation(len(data))[:svm_sub_sample_size]
        predictor.fit(data[_idx], pred_labels[_idx])
    else:
        predictor.fit(data, pred_labels)
    print(f'Finish svm trainign')

    d_indices_list, d_data_list, centroid_ids_list = separate_by_centroids(centroids, pred_labels, data_indices, data)
    filtered_centroids = centroids
    # del centroids
    
    if db_centroids is None:
        db_centroids = [filtered_centroids]
    else:
        db_centroids.append(filtered_centroids)
    updated_db_centroids_size = current_db_centroids_size + len(filtered_centroids)
    search_pair = (current_db_centroids_size, updated_db_centroids_size)

    intermediate_centroids_size = updated_db_centroids_size
    for i, (_d_indices, _data) in enumerate(zip(d_indices_list, d_data_list)):
        if len(_data) > max_samples_cluster * 2 and max_depth > 0:
            sub_k = min(k, (len(_data) // max_samples_cluster))
            next_level_database, intermediate_centroids_size = multi_layer_kmeans_gpus_svm(
                _d_indices, _data, sub_k, max_samples_cluster, retain_max_samples, 
                ngpu, fp16, db_centroids, mem, niter, nredo,
                gpu_resources=gpu_resources,
                max_depth=max_depth - 1,
                cluster_same_size=cluster_same_size,
                current_db_centroids_size=intermediate_centroids_size,
                svm_kernel=svm_kernel,
                svm_c=svm_c,
                svm_sub_sample_size=svm_sub_sample_size
            )
            db_centroids = next_level_database["db_centroids"]
            sub_pair = next_level_database["start_pair"]
            sub_db_dict = next_level_database['db_dict']
            sub_predictor = next_level_database['predictor']
            db_dict[i] = (sub_pair, sub_db_dict, sub_predictor)
        else:
            db_dict[i] = _d_indices[np.random.permutation(len(_d_indices))[:retain_max_samples]]

    if max_depth <= 0:
        print(f'Max depth reach: {len(data)=}')
    
    print(f'finish sub branch {max_depth=}, {intermediate_centroids_size=}', end='\r')
    output_dict = {
        "predictor": predictor,
        "db_centroids": db_centroids,
        "start_pair": search_pair,
        "db_dict": db_dict
    }
    if db_centroids_empty:
        print(f'Finally post processing, concatenate db_centroids')
        output_dict['db_centroids'] = np.concatenate(db_centroids, 0)
        assert len(output_dict['db_centroids']) == intermediate_centroids_size, f"{len(output_dict['db_centroids'])=} != {intermediate_centroids_size=}"
        return output_dict
    else:
        return output_dict, intermediate_centroids_size


class PiecewiseLinearDiscreteFn:
    """Piecewise linear function. Can be configured with a string."""

    def __init__(self, pieces: Sequence[Tuple[int, float]]):
        assert pieces == sorted(
            pieces
        ), f"PiecewiseLinearDiscreteFn configuration should be sorted, received: {pieces}"

        self.pieces = pieces

    def __call__(self, x: int) -> float:
        for i, (x_a, y_a) in enumerate(self.pieces[:-1]):
            x_b, y_b = self.pieces[i + 1]
            if x_a <= x <= x_b:
                # return y_a + (x - x_a) * (y_b - y_a) / (x_b - x_a)
                return int(y_a)

        return int(self.pieces[-1][1])

    @staticmethod
    def from_string(configuration: str) -> "PiecewiseLinearDiscreteFn":
        """
        Parse the configuration of lambda coefficient (for scheduling).
        x = "3"                  # lambda will be a constant equal to x
        x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                                 # to 0 during the first 1000 iterations
        x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                                 # iterations, then will linearly increase to 1 until iteration 2000
        """
        if isinstance(configuration, float):
            return PiecewiseLinearDiscreteFn([(0, configuration)])

        try:
            parts = configuration.split(",")
            if len(parts) == 1:
                v = float(configuration)
                return PiecewiseLinearDiscreteFn([(0, v)])

            split = [s.split(":") for s in parts]
            pieces = [(int(t), float(v)) for t, v in split]
            return PiecewiseLinearDiscreteFn(pieces)
        except Exception:
            raise ValueError(
                f"Invalid PiecewiseLinearDiscreteFn configuration: {configuration!r}"
            )

    @staticmethod
    def one() -> "PiecewiseLinearDiscreteFn":
        return PiecewiseLinearDiscreteFn([(0, 1.0)])
        

class _DbNNClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, data_size, gpu_id=None, **kwargs):
        super(_DbNNClassifier, self).__init__()
        self.data_size = data_size
        self.gpu_id = gpu_id
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        size_to_layers = kwargs.get("size_to_layers", None)
        if size_to_layers is not None and size_to_layers != "":
            # self.n_layers = kwargs["size_to_layers"](data_size)
            self.n_layers = PiecewiseLinearDiscreteFn.from_string(size_to_layers)(data_size)
            # print(f'{self.n_layers=}')
        else:
            self.n_layers = kwargs.get('n_layers', 0)
        if self.n_layers == 0:
            self.ffn = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, self.hidden_dim)]
            for l in range(self.n_layers):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_dim, out_dim))
            self.ffn = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.ffn(x)
        x = self.logsoftmax(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = self.ffn(x).argmax(-1)
            x = x.numpy()
            return x
    
    def fit(self, data, labels, **kwargs):
        lr = kwargs.get("lr", 0.0001)
        batch_size = kwargs.get("batch_size", len(data))
        # epoch = kwargs.get("epoch", 1)
        train_steps = kwargs.get("train_steps", 100)
        stop_acc = kwargs.get("stop_acc", -1)
        momentum = kwargs.get("acc_momentum", 0.8)

        self.cuda(device=self.gpu_id)

        loss_fn = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        moving_acc = 0
        steps = 0
        while steps < train_steps:
            for i in range(0, len(data), batch_size):
                _input = torch.from_numpy(data[i:i + batch_size])
                _target = torch.from_numpy(labels[i:i + batch_size]).long()
                _input, _target = move_to_cuda((_input, _target), device=f"cuda:{self.gpu_id}" if self.gpu_id is not None else None)

                optimizer.zero_grad()
                _outputs = self(_input)
                loss = loss_fn(_outputs, _target)
                loss.backward()
                optimizer.step()
                if stop_acc > 0:
                    _acc = (_outputs.argmax(-1) == _target).float().mean()
                    # moving_acc = momentum * _acc + (1 - momentum) * moving_acc
                    moving_acc = (1 - momentum) * _acc + momentum * moving_acc
                steps += 1
            if moving_acc > stop_acc > 0:
                break

        optimizer.zero_grad(set_to_none=True)
        if moving_acc < stop_acc:
            print(f'{moving_acc=} < {stop_acc=} after {train_steps=}, {data.shape=}')
        self.cpu()
        # clean up
        del optimizer, loss_fn


def build_db_predictor(data, pred_labels, n_classes, predictor_type, gpu_id=None, **kwargs):
    if predictor_type == "svm":
        from sklearn import svm
        data = data.astype("float32")
        predictor = svm.SVC(kernel=kwargs.get("svm_kernel", 'rbf'), C=kwargs.get("svm_c", 1))
        svm_sub_sample_size = kwargs.get("svm_sub_sample_size", 0)
        if svm_sub_sample_size > 0:
            _idx = np.random.permutation(len(data))[:svm_sub_sample_size]
            predictor.fit(data[_idx], pred_labels[_idx])
        else:
            predictor.fit(data, pred_labels)
        return predictor
    elif predictor_type == 'linear_nn':
        predictor = _DbNNClassifier(data.shape[-1], n_classes, data.shape[0], gpu_id=gpu_id, **kwargs)
        predictor.fit(data, pred_labels, **kwargs)
    return predictor


def multi_layer_kmeans_gpus_predictor(
    data_indices, 
    data, 
    k, 
    max_samples_cluster, 
    retain_max_samples, 
    ngpu, 
    fp16=False, 
    mem=5*1024*1024*1024,
    niter=20,
    nredo=1,
    gpu_resources=None,
    max_depth=50,
    cluster_same_size=False,
    # predictor args
    **kwargs,
):
    # FIXME: this is taking lots of memory gradually, why?

    db_dict = {}
    # db_centroids_empty = db_centroids is None   # root level
    # assert retain_max_samples <= max_samples_cluster
    # kmeans

    gpu_resources, index, centroids = faiss_kmeans_gpus(
        data, k, ngpu, 
        resources=gpu_resources,
        fp16=fp16, mem=mem, niter=niter, nredo=nredo
    )
    if cluster_same_size:
        distances, topk = index.search(data, k)
        del index
        pred_labels, centroids = same_size_cluster_rebalance(data, distances, topk, k, centroids)
        del distances, topk
    else:
        pred_labels = index.search(data, 1)[1][:, 0] # # we want to see 1 nearest neighbors
        del index

    predictor = build_db_predictor(data, pred_labels, k, **kwargs)

    d_indices_list, d_data_list, centroid_ids_list = separate_by_centroids(centroids, pred_labels, data_indices, data)
    del centroids

    for i, (_d_indices, _data) in enumerate(zip(d_indices_list, d_data_list)):
        if len(_data) > max_samples_cluster * 2 and max_depth > 0:
            sub_k = min(k, (len(_data) // max_samples_cluster))
            next_level_database = multi_layer_kmeans_gpus_predictor(
                _d_indices, _data, sub_k, 
                max_samples_cluster, retain_max_samples, 
                ngpu, fp16, mem, niter, nredo,
                gpu_resources=gpu_resources,
                max_depth=max_depth - 1,
                cluster_same_size=cluster_same_size,
                **kwargs
            )
            sub_db_dict = next_level_database['db_dict']
            sub_predictor = next_level_database['predictor']
            db_dict[i] = (sub_db_dict, sub_predictor)
        else:
            db_dict[i] = _d_indices[np.random.permutation(len(_d_indices))[:retain_max_samples]]

    if max_depth <= 0:
        print(f'Max depth reach: {len(data)=}')
    
    print(f'finish sub branch {max_depth=}, ', end='\r')
    output_dict = {
        "db_dict": db_dict,
        "predictor": predictor,
    }
    return output_dict


def multi_layer_kmeans_gpus_predictor_bfs(
    # data_indices, 
    # data, 
    load_ids_n_embeds,
    k, 
    max_samples_cluster, 
    retain_max_samples, 
    ngpu, 
    fp16=False, 
    mem=5*1024*1024*1024,
    niter=20,
    nredo=1,
    decode_block_size=2 ** 15,
    gpu_resources=None,
    max_depth=50,
    cluster_same_size=False,
    # predictor args
    **kwargs,
):
    """
    jobs: kmeans, predictor, split_datas
    
    # level 0:
    0 kmeans predictor partitions
    # level 1:
    # 0x0: kmeans predictor
    # 0x1: kmeans predictor
    # level 2:
    # 00x0: kmeans predictor
    # 00x1: kmeans predictor
    # 01x2: kmeans predictor
    # 01x3: kmeans predictor
    # level 3:
    # 000x0: kmeans predictor
    # 000x1: kmeans predictor
    # 001x0: kmeans predictor
    # 001x1: kmeans predictor
    # 012x0: kmeans predictor
    # 012x1: kmeans predictor

    codes:
    predictors = {}
    job_details = {}

    job_details[0] = prepare_inputs_for_first(data)
    for level in range(0, upper(log_k(n))):
        partition_dict = {}
        for j, job for enumerate(job_details[level]):
            predictor, data_partitions = run_job(job)
            predictors[job.key] = predictor
            partition_dict[job.key] = data_partitions   # pdict[0] = {00: d0, 01: d1}
        job_details[level] = prepare_next_job_batches(level, partition_dict)
                            # jd[0] = [(00: d0), (01: d1)]
                            # jd[1] = [(000: d0), (001: d1), (010: d2), (011: d3)]
    """
    import math
    import gc
    predictors = {}
    job_details = {}
    buckets = {}

    data_indices, data = load_ids_n_embeds()

    max_level = math.ceil(math.log(len(data))) + 1
    print(f'Max level: {max_level}')

    gpu_resources = get_faiss_gpu_resources(ngpu, mem=mem)
    def run_job(job_detail, gpu_resources):
        job_key = job_detail['key']
        job_data = job_detail['data']
        job_indices = job_detail['indices']
        job_k = job_detail['k']

        # build kmeans
        gpu_resources, index, centroids = faiss_kmeans_gpus(
            job_data, job_k, ngpu, 
            resources=gpu_resources,
            fp16=fp16, 
            mem=mem, 
            niter=niter, 
            nredo=nredo,
            decode_block_size=decode_block_size,
        )
        if cluster_same_size:
            distances, topk = index.search(job_data, job_k)
            del index
            pred_labels = rebalance_cluster_assignments(job_data.shape[0], distances, topk, job_k)
            del distances, topk
        else:
            pred_labels = index.search(job_data, 1)[1][:, 0] # # we want to see 1 nearest neighbors
            del index
        del centroids

        d_indices_list, d_data_list, centroid_ids_list = separate_by_centroid_pred_labels(job_k, pred_labels, job_indices, job_data)

        predictor = build_db_predictor(job_data, pred_labels, job_k, **kwargs)
        job_outcome = {
            "predictor": predictor,
            "partitions": {
                f'{job_key}{j}': {
                    "d_idx": _d_idx, 
                    "d_data": _d_data,
                    "sub_k": min(job_k, (len(_d_data) // max_samples_cluster))
                }
                for j, (_d_idx, _d_data) in enumerate(zip(d_indices_list, d_data_list))
            },
        }
        return gpu_resources, job_outcome
    
    def _build_kmeans_batch(job_details, gpu_resources):
        d_indices_batch = []
        d_data_batch = []
        pred_labels_batch = []
        for j, job_detail in enumerate(job_details):
            # job_key = job_detail['key']
            job_data = job_detail['data']
            job_indices = job_detail['indices']
            job_k = job_detail['k']
            # TODO: profile: faiss_kmeans_gpus should have minimal gpu idle time
            gpu_resources, index, centroids = faiss_kmeans_gpus(
                job_data, job_k, ngpu, 
                resources=gpu_resources,
                fp16=fp16, 
                mem=mem, 
                niter=niter,
                nredo=nredo,
                decode_block_size=decode_block_size,
            )
            # TODO: profile: is index.search take most gpu idle time?
            if cluster_same_size:
                distances, topk = index.search(job_data, job_k)
                del index
                # TODO: profile: is this one take most gpu idle time?
                pred_labels = rebalance_cluster_assignments(job_data.shape[0], distances, topk, job_k)
                del distances, topk
            else:
                pred_labels = index.search(job_data, 1)[1][:, 0] # # we want to see 1 nearest neighbors
                del index
            del centroids
            # TODO: profile: is separate_by_centroids take most gpu idle time?
            d_indices_list, d_data_list, centroid_ids_list = separate_by_centroid_pred_labels(job_k, pred_labels, job_indices, job_data)
            d_indices_batch.append(d_indices_list)
            d_data_batch.append(d_data_list)
            pred_labels_batch.append(pred_labels)
        return d_indices_batch, d_data_batch, pred_labels_batch
    
    def _build_kmeans_batch_multiprocessing(job_details, gpu_resources):
        """
        Run (iterative faiss_kmeans_gpus + index.search) => (distances, topk) | (pred_labels)
        if cluster_same_size:
            run-parallel rebalance_cluster_assignments with multi-processing
        """
        d_indices_batch = []
        d_data_batch = []
        pred_labels_batch = []
        index_search_results = []
        # TODO: run iterative kmeans
        for j, job_detail in enumerate(job_details):
            # job_key = job_detail['key']
            job_data = job_detail['data']
            job_indices = job_detail['indices']
            job_k = job_detail['k']
            # TODO: profile: faiss_kmeans_gpus should have minimal gpu idle time
            gpu_resources, index, centroids = faiss_kmeans_gpus(
                job_data, job_k, ngpu, 
                resources=gpu_resources,
                fp16=fp16, 
                mem=mem, 
                niter=niter,
                nredo=nredo,
                decode_block_size=decode_block_size,
            )
            if cluster_same_size:
                distances, topk = index.search(job_data, job_k)
                del index
                index_search_results.append((distances, topk))
            else:
                pred_labels = index.search(job_data, 1)[1][:, 0] # # we want to see 1 nearest neighbors
                index_search_results.append(pred_labels)
                del index
            del centroids
        # TODO: run multi-processing rebalance/cluster assignments
        if cluster_same_size:
            # multi-processing, replace index_search_results with pred_labels
            with multiprocessing.Manager() as manager:
                def _rebalance_assign(_data_size, _distances, _topk, _job_k, _pindex, _result_assign):
                    _pred_labels = rebalance_cluster_assignments(_data_size, _distances, _topk, _job_k)
                    _result_assign[_pindex] = _pred_labels
                
                nprocs = ngpu
                for j in range(0, len(job_details), nprocs):
                    job_sub_batch = job_details[j:j + nprocs]
                    search_result_batch = index_search_results[j:j + nprocs]
                    job_sub_len = len(job_sub_batch)

                    processes = []
                    sub_proc_dict = manager.dict()
                    for _pindex, (job_detail, search_result) in enumerate(zip(job_sub_batch, search_result_batch)):
                        _job_data = job_detail['data']
                        _job_k = job_detail['k']
                        _distances, _topk = search_result
                        _p = multiprocessing.Process(
                            target=_rebalance_assign, 
                            args=(_job_data.shape[0], _distances, _topk, _job_k, _pindex, sub_proc_dict), 
                            daemon=True)
                        _p.start()
                        processes.append(_p)
                    for _p in processes:
                        _p.join()
                    for _jj in range(job_sub_len):
                        index_search_results[j + _jj] = sub_proc_dict[_jj]

            assert all(not isinstance(x, tuple) for x in index_search_results)

        for j, job_detail in enumerate(job_details):
            job_k = job_detail['k']
            job_data = job_detail['data']
            job_indices = job_detail['indices']
            pred_labels = index_search_results[j]
            d_indices_list, d_data_list, _ = separate_by_centroid_pred_labels(job_k, pred_labels, job_indices, job_data)
            d_indices_batch.append(d_indices_list)
            d_data_batch.append(d_data_list)
            pred_labels_batch.append(pred_labels)
        return d_indices_batch, d_data_batch, pred_labels_batch
    
    def _build_db_predictor_batch(job_details, d_indices_batch, d_data_batch, pred_labels_batch, gpu_resources):
        # TODO: try to parallel predictor learning
        assert len(job_details) == len(pred_labels_batch)
        predictor_batch = []
        for j, (job_detail, pred_labels) in enumerate(zip(job_details, pred_labels_batch)):
            job_data = job_detail['data']
            job_k = job_detail['k']
            predictor = build_db_predictor(job_data, pred_labels, job_k, **kwargs)
            predictor_batch.append(predictor)
        return predictor_batch
    
    def _build_db_predictor_batch_multithread(job_details, d_indices_batch, d_data_batch, pred_labels_batch, gpu_resources):
        # TODO: multi-thread
        assert len(job_details) == len(pred_labels_batch)
        if ngpu < 2:
            return _build_db_predictor_batch(job_details, d_indices_batch, d_data_batch, pred_labels_batch, gpu_resources)
        predictor_batch = []

        def _subthread_build_db_predictor(_id, _job_detail, _pred_labels, _sub_predictor_assign, _gpu_id):
            __predictor = build_db_predictor(_job_detail['data'], _pred_labels, _job_detail['k'], gpu_id=_gpu_id, **kwargs)
            _sub_predictor_assign[_id] = __predictor

        for j in range(0, len(job_details), ngpu):
            job_sub_batch = job_details[j:j + ngpu]
            pred_labels_sub_batch = pred_labels_batch[j:j + ngpu]
            job_sub_len = len(job_sub_batch)

            _threads = []
            sub_predictor_assign = [None for i in range(job_sub_len)]
            for _pindex, (job_detail, pred_labels) in enumerate(zip(job_sub_batch, pred_labels_sub_batch)):
                _gpu_id = _pindex
                p = threading.Thread(
                    target=_subthread_build_db_predictor, 
                    args=(_pindex, job_detail, pred_labels, sub_predictor_assign, _gpu_id), daemon=True)
                p.start()
                _threads.append(p)
            for p in _threads:
                p.join()
            assert not any(x is None for x in sub_predictor_assign), f'{sub_predictor_assign}'
            predictor_batch.extend(sub_predictor_assign)
        return predictor_batch
    
    def run_job_batch(job_details, gpu_resources):
        # TODO: try to parallel as much as possible
        cur_time = time.perf_counter()
        # d_indices_batch, d_data_batch, pred_labels_batch = _build_kmeans_batch(job_details, gpu_resources)
        d_indices_batch, d_data_batch, pred_labels_batch = _build_kmeans_batch_multiprocessing(job_details, gpu_resources)
        km_batch_time = time.perf_counter() - cur_time
        print(f'{km_batch_time=}')

        # predictor_batch = _build_db_predictor_batch(job_details, d_indices_batch, d_data_batch, pred_labels_batch, gpu_resources)
        # on valid set
        # 58.54059834207874
        cur_time = time.perf_counter()
        predictor_batch = _build_db_predictor_batch_multithread(job_details, d_indices_batch, d_data_batch, pred_labels_batch, gpu_resources)
        predictor_time = time.perf_counter() - cur_time
        print(f'{predictor_time=}')

        # 46 secs (4gpu) / (2gpu still the same)
        job_outcomes = [{
            "key": job['key'],
            "predictor": predictor,
            "partitions": {
                f'{job["key"]}{j}': {
                    "d_idx": _d_idx, 
                    "d_data": _d_data,
                    "sub_k": min(job["k"], (len(_d_data) // max_samples_cluster))
                }
                for j, (_d_idx, _d_data) in enumerate(zip(d_indices_list, d_data_list))
            },
        } for b, (job, predictor, d_indices_list, d_data_list) in enumerate(zip(
            job_details, predictor_batch, d_indices_batch, d_data_batch
        ))
        ]
        return gpu_resources, job_outcomes
    
    def prepare_next_job_batches(partition_dict, level):
        next_job_batches = []
        for k, v in partition_dict.items():
            for jk, partition in v.items():
                if len(partition["d_idx"]) > max_samples_cluster * 2 and level < max_level:
                    new_job = {
                        "key": jk,
                        "indices": partition["d_idx"],
                        "data": partition["d_data"],
                        "k": partition["sub_k"]
                    }
                    next_job_batches.append(new_job)
                else:
                    # buckets[jk] = partition["d_idx"]
                    buckets[jk] = partition["d_idx"][np.random.permutation(len(partition["d_idx"]))[:retain_max_samples]]
        return next_job_batches
    
    # TODO: assign first job_details
    job_details[0] = [{
        "key": "0",
        "indices": data_indices,
        "data": data,
        "k": k
    }]
    # # main for loop
    # for level in range(0, max_level):
    #     partition_dict = {}
    #     for j, job in enumerate(job_details[level]):
    #         gpu_resources, job_out = run_job(job, gpu_resources)
    #         job_key = job['key']
    #         predictors[job_key] = job_out['predictor']
    #         partition_dict[job_key] = job_out['partitions']
    #     # prepare next job batches
    #     job_details[level + 1] = prepare_next_job_batches(partition_dict, level)
        
    # main for loop
    for level in range(0, max_level):
        partition_dict = {}
        gpu_resources, job_outcomes = run_job_batch(job_details[level], gpu_resources)
        for job_out in job_outcomes:
            predictors[job_out['key']] = job_out['predictor']
            partition_dict[job_out['key']] = job_out['partitions']
        # prepare next job batches
        job_details[level + 1] = prepare_next_job_batches(partition_dict, level)
        print(f'{level=} finished ==================================')

        del job_details[level]
        if level == 0:
            print(f'delete data and data_indices')
            del data, data_indices
        del partition_dict
        gc.collect()

    database = {
        "predictors": predictors,
        "buckets": buckets,
    }
    return database


def multi_layer_db_search_single_predictor_bfs(data, kmeans_database, get_search_tree=False, search_tree=None):
    predictors = kmeans_database['predictors']
    buckets = kmeans_database['buckets']
    if len(data.shape) == 1:
        data = data[None, :]
    assert data.shape[0] == 1, f'only support single query: {data.shape=}'
    cur_key = "0"

    out_bucket = None
    safe_check = 0
    while True:
        _predictor = predictors[cur_key]
        preds = _predictor.predict(data)[0]
        next_key = f'{cur_key}{preds}'

        if next_key not in predictors:
            # expect next_key in buckets
            out_bucket = buckets[next_key]
            break
        else:
            cur_key = next_key
        safe_check += 1
        if safe_check > 100000:
            print(f'Suspect circular refeence: {next_key}')
            break
    return out_bucket, next_key


def separate_by_centroids(centroids, pred_labels, _km_indices, _km_data):
    d_indices_list = []
    d_data_list = []
    centroid_ids_list = []
    for i in range(len(centroids)):
        _is_label_c = pred_labels == i
        _d_indices = _km_indices[_is_label_c]
        _data = _km_data[_is_label_c]
        d_indices_list.append(_d_indices)
        d_data_list.append(_data)
        centroid_ids_list.append(i)
    return d_indices_list, d_data_list, centroid_ids_list


def separate_by_centroid_pred_labels(km_k, pred_labels, _km_indices, _km_data):
    d_indices_list = []
    d_data_list = []
    centroid_ids_list = []
    for i in range(km_k):
        _is_label_c = pred_labels == i
        _d_indices = _km_indices[_is_label_c]
        _data = _km_data[_is_label_c]
        d_indices_list.append(_d_indices)
        d_data_list.append(_data)
        centroid_ids_list.append(i)
    return d_indices_list, d_data_list, centroid_ids_list


def filter_centroids(centroids, pred_labels, _km_indices, _km_data):
    d_indices_list = []
    d_data_list = []
    centroid_ids_list = []
    for i in range(len(centroids)):
        _is_label_c = pred_labels == i
        if _is_label_c.astype(int).sum() > 0:
            _d_indices = _km_indices[_is_label_c]
            _data = _km_data[_is_label_c]
            d_indices_list.append(_d_indices)
            d_data_list.append(_data)
            centroid_ids_list.append(i)
    return d_indices_list, d_data_list, centroid_ids_list


def save_kmeans_database(filename, kmeans_database):
    import pickle
    if not filename.endswith(".npy"):
        filename = filename + ".npy"
    with open(filename, 'wb') as f:
        np.save(f, kmeans_database)


def load_kmeans_database(filename):
    with open(filename, 'rb') as f:
        out_dict = np.load(f, allow_pickle=True)
        if not isinstance(out_dict, dict):
            out_dict = out_dict[()]
            assert isinstance(out_dict, dict), f'{type(out_dict)=}'
        # db_centroids, start_pair, db_dict = out_dict["db_centroids"], out_dict["start_pair"], out_dict["db_dict"]
    return out_dict


def _inspect_km_db_pockets(db_dict, pockets=None):
    if pockets is None:
        pockets = []
    for k, v in db_dict.items():
        if isinstance(v, tuple):
            sub_dict = v[1]
            _inspect_km_db_pockets(sub_dict, pockets)
        else:
            pockets.append(v)
    return pockets


def _inspect_km_db_pockets_predictor(db_dict, pockets=None):
    if pockets is None:
        pockets = []
    for k, v in db_dict.items():
        if isinstance(v, tuple):
            sub_dict = v[0]
            _inspect_km_db_pockets_predictor(sub_dict, pockets)
        else:
            pockets.append(v)
    return pockets


def _inspect_km_db_depths(db_dict, current_depth=0):
    max_depth = current_depth
    min_depth = current_depth
    fixed_min_depth = None
    # fixed_max_depth = current_depth
    for k, v in db_dict.items():
        if isinstance(v, tuple):
            _max_depth, _min_depth = _inspect_km_db_depths(v[1], current_depth=current_depth + 1)
            max_depth = max(max_depth, _max_depth)
            min_depth = min(min_depth, _min_depth)
        else:
            fixed_min_depth = current_depth
    return max_depth, (min_depth if fixed_min_depth is None else fixed_min_depth)


SPM_UNDERSCORE = "▁"
def _mbart_decode(x):
    return x.replace(" ", "").replace(SPM_UNDERSCORE, " ").strip()


def _inspect_km_db_with_txt(db_dict, txt_path, lines=None, current_depth=0, edit_distances=None, sub_dict_index=1):
    import editdistance
    if lines is None:
        # load 
        with open(txt_path, 'r', encoding='utf-8', errors='surrogateescape') as f:
            lines = f.read().splitlines()
    if edit_distances is None:
        edit_distances = []
    for k, v in db_dict.items():
        if isinstance(v, tuple):
            sub_dict = v[sub_dict_index]
            _inspect_km_db_with_txt(sub_dict, txt_path, lines, current_depth + 1, edit_distances, sub_dict_index=sub_dict_index)
        else:
            v_list = v.tolist()
            assert len(set(v_list)) == len(v_list), f'{len(set(v_list))=} != {len(v_list)=}, {v_list}'
            leaf_lines = [lines[i] for i in v_list]
            leaf_tokens = [x.split(" ") for x in leaf_lines]
            # 
            len_leaf_tokens = len(leaf_tokens)
            avg_distance = sum(
                editdistance.eval(leaf_tokens[i], leaf_tokens[j])
                for i in range(len_leaf_tokens - 1)
                for j in range(i + 1, len_leaf_tokens)
            ) / ((len_leaf_tokens ** 2 -  len_leaf_tokens)/2)
            sent_lens = [len(x) for x in leaf_tokens]
            mean_sent_lens = sum(sent_lens) / (1.0 * len(sent_lens))
            print(' ' * current_depth + f'[{current_depth}]/{k}: avg_distance: {avg_distance} ({avg_distance / mean_sent_lens} per sent-length) / {len_leaf_tokens}')

            if np.random.uniform() < 0.2 and avg_distance > 10:
                leaf_sents = [_mbart_decode(x) for x in leaf_lines]
                for j, l in enumerate(leaf_sents):
                    # print(' ' * (current_depth + 2) + f'{j}: {l}')
                    print(f'{l}')
                _ = input('continued?')

            edit_distances.append(avg_distance)
    return edit_distances


def inspect_kmeans_db(db, txt_path=None):
    db_centroids = db['db_centroids']
    db_dict = db['db_dict']
    print(f'centroids: {db_centroids.shape}')
    pockets = _inspect_km_db_pockets(db_dict)
    len_pockets = [len(x) for x in pockets]
    max_pock = max(len_pockets)
    min_pock = min(len_pockets)
    mean_pock = sum(len_pockets) / (1.0 * len(len_pockets))
    print(f'pockets: {len(pockets)=}, {max_pock=}, {min_pock=}, {mean_pock=}')
    n_zero_pock = len([x for x in len_pockets if x == 0])
    n_max_pock = len([x for x in len_pockets if x == max_pock])
    print(f'{n_zero_pock=}, {n_max_pock=}')
    max_depth, min_depth = _inspect_km_db_depths(db_dict)
    print(f'{max_depth=} {min_depth=}')
    print(f'Inspect with txt files')
    if txt_path is not None:
        edit_distances = _inspect_km_db_with_txt(db_dict, txt_path)
        max_mean_edit_distance = max(edit_distances)
        min_mean_edit_distance = min(edit_distances)
        mean_mean_edit_distance = sum(edit_distances) / len(edit_distances)
        print(f'{max_mean_edit_distance=}, {min_mean_edit_distance=}, {mean_mean_edit_distance=}')


def inspect_kmeans_db_predictor(db, txt_path=None):
    # db_centroids = db['db_centroids']
    db_dict = db['db_dict']
    # print(f'centroids: {db_centroids.shape}')
    pockets = _inspect_km_db_pockets_predictor(db_dict)
    len_pockets = [len(x) for x in pockets]
    max_pock = max(len_pockets)
    min_pock = min(len_pockets)
    mean_pock = sum(len_pockets) / (1.0 * len(len_pockets))
    print(f'pockets: {len(pockets)=}, {max_pock=}, {min_pock=}, {mean_pock=}')
    n_zero_pock = len([x for x in len_pockets if x == 0])
    n_max_pock = len([x for x in len_pockets if x == max_pock])
    print(f'{n_zero_pock=}, {n_max_pock=}')
    # max_depth, min_depth = _inspect_km_db_depths(db_dict)
    # print(f'{max_depth=} {min_depth=}')
    print(f'Inspect with txt files')
    if txt_path is not None:
        edit_distances = _inspect_km_db_with_txt(db_dict, txt_path, sub_dict_index=0)
        max_mean_edit_distance = max(edit_distances)
        min_mean_edit_distance = min(edit_distances)
        mean_mean_edit_distance = sum(edit_distances) / len(edit_distances)
        print(f'{max_mean_edit_distance=}, {min_mean_edit_distance=}, {mean_mean_edit_distance=}')


