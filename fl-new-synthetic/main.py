import torch
from torch.utils.data import DataLoader
from config import *
from dataset.dataset import *
from statistic.collect_stat import CollectStatistics
from util.util import *
import numpy as np
import random
import copy
from model.model import Model
# from util.util import DatasetSplit
# print(dataset_file_path)
# import json
def include_worker_gradient(worker_index, gradient, global_loss, local_loss, error, method = "None"):
    if method == "None":
        return False
    elif method == "All":
        return True
    elif method == "Ada":
        if np.abs(global_loss - local_loss) < error:
            return True
        else:
            return False
#     return True
if device.type != 'cpu':
    torch.cuda.set_device(device)

if __name__ == "__main__":
    stat = CollectStatistics(results_file_name=results_file)

    for seed in simulations:

        random.seed(seed)
        np.random.seed(seed)  # numpy
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        torch.backends.cudnn.deterministic = True  # cudnn
        
        data_train, data_test = load_data(dataset, dataset_file_path, 'cpu', config_dict)
#         num_examples = len(data_train)
#         num_examples_subset_client = int(num_examples * subset_size_client)
#         num_examples_subset_worker = num_examples - num_examples_subset_client
#         data_train_client, data_train_worker = torch.utils.data.random_split(
#     data_train, [num_examples_subset_client, num_examples_subset_worker]
# )
#         data_train_loader = DataLoader(data_train, batch_size=batch_size_eval, shuffle=True, num_workers=0)
# #         data_train_client_loader = DataLoader(data_train_client, batch_size=batch_size_eval, shuffle=True, num_workers=0)
# #         data_train_worker_loader = DataLoader(data_train_worker, batch_size=batch_size_eval, shuffle=True, num_workers=0)
#         data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)
#         dict_users = split_data(dataset, data_train_client, n_nodes)
#         dict_workers = split_data(dataset, data_train_worker, n_workers,typ = 'worker')  # Separate dictionary for workers
        if n_nodes is None:
            n_nodes = len(dict_users)
#         print(max(data_train_client.indices))
#         print(max(data_train_worker.indices))
#         print(len(data_train_client.indices))
# #         node_sampler = NodeSampler(n_nodes, permutation=use_permute)

        if iters_warmup > 0:
            step_size = step_size_warmup
            warming_up = True
        else:
            step_size = step_size_local
            warming_up = False
        print(step_size_local)
        model = Model(seed, step_size_local, model_name=model_name, device=device, flatten_weight=True,
                      pretrained_model_file=load_model_file)

#         train_loader_list = []
#         dataiter_list = []
#         for n in range(n_nodes):
#             train_loader_list.append(
#                 DataLoader(DatasetSplit(data_train_client, dict_users[n]), batch_size=batch_size_train, shuffle=True))
#             dataiter_list.append(iter(train_loader_list[n]))
#         worker_train_loader_list = []
#         dataiter_list_w = []
#         for w in range(n_workers):
#             worker_train_loader_list.append(
#                 DataLoader(DatasetSplit(data_train_worker, dict_workers[w]), batch_size=batch_size_train, shuffle=True))
#             dataiter_list_w.append(iter(worker_train_loader_list[w]))
        train_loader_list, dataiter_list, data_train_loader, data_test_loader, worker_train_loader_list, dataiter_list_w = create_data_loaders(data_train, data_test, batch_size_train = batch_size_train, batch_size_test = batch_size_eval)

        def sample_minibatch(n):
            try:
                images, labels = dataiter_list[n].next()
                if len(images) < batch_size_train:
                    dataiter_list[n] = iter(train_loader_list[n])
                    images, labels = dataiter_list[n].next()
            except StopIteration:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = dataiter_list[n].next()

            return images, labels

        def sample_full_batch(n):
            images = []
            labels = []
            for i in range(len(train_loader_list[n].dataset)):
                images.append(train_loader_list[n].dataset[i][0])

                l = train_loader_list[n].dataset[i][1]
                if not isinstance(l, torch.Tensor):
                    l = torch.as_tensor(l)
                labels.append(l)

            return torch.stack(images), torch.stack(labels)
        def sample_minibatch_w(n):
            try:
                images, labels = dataiter_list_w[n].next()
                if len(images) < batch_size_train:
                    dataiter_list_w[n] = iter(worker_train_loader_list[n])
                    images, labels = dataiter_list_w[n].next()
            except StopIteration:
                dataiter_list_w[n] = iter(worker_train_loader_list[n])
                images, labels = dataiter_list_w[n].next()

            return images, labels

#         def sample_full_batch_w(n):
#             images = []
#             labels = []
#             for i in range(len(worker_train_loader_list[n].dataset)):
#                 images.append(worker_train_loader_list[n].dataset[i][0])

#                 l = train_loader_list[n].dataset[i][1]
#                 if not isinstance(l, torch.Tensor):
#                     l = torch.as_tensor(l)
#                 labels.append(l)
        def sample_full_batch_w(n):
            images_list, labels_list = [], []

            for images, labels in worker_train_loader_list[n]:
                images_list.append(images)
                labels_list.append(labels)

            # Concatenate all images and labels
            images_full_batch = torch.cat(images_list, dim=0)
            labels_full_batch = torch.cat(labels_list, dim=0)

            return images_full_batch, labels_full_batch

            return torch.stack(images), torch.stack(labels)

        w_global = model.get_weight()   # Get initial weight

        num_iter = 0
        last_output = 0
        last_amplify = 0
        last_save_latest = 0
        last_save_checkpoint = 0

#         if periodic_availability:
#             # For periodic node connectivity
#             active_nodes = np.zeros(n_nodes)
#             ranges_inactive = []
#             for n in range(0, n_nodes):
#                 for i in range(0, 5):
#                     if int(np.round(i * n_nodes / 5)) <= n < int(np.round((i + 1) * n_nodes / 5)):
#                         ranges_inactive.append(i)

#             active_nodes[0:int(np.round(n_nodes / 5))] = 1
#             offset = int(active_rounds * np.random.rand())  # To avoid different simulation runs from being synchronized

#             time_until_switch = []
#             for n in range(n_nodes):
#                 if active_nodes[n] == True:
#                     time_until_switch.append(active_rounds - offset)
#                 else:
#                     time_until_switch.append(active_rounds * ranges_inactive[n] - offset)
#         else:
        # Full connectivity
#         active_nodes = np.ones(n_nodes)

        if use_global_update and not warming_up:
            w_global_prev_for_add_update = copy.deepcopy(w_global.to('cpu'))

        while True:
#             if warming_up and num_iter >= iters_warmup:
#                 step_size = step_size_local
#                 warming_up = False

#                 model = Model(seed, step_size, model_name=model_name, device=device, flatten_weight=True,
#                               pretrained_model_file=load_model_file)

#                 if use_global_update:
#                     w_global_prev_for_add_update = copy.deepcopy(w_global.to('cpu'))

#                 last_amplify = num_iter

            print('seed', seed,'  iteration', num_iter)

#             if wait_for_all and not warming_up:
            node_subset = [n for n in range(0, n_nodes)]
#             else:
#                 node_sample_set = [n for n in range(0, n_nodes) if active_nodes[n] == True]
#                 node_subset = node_sampler.sample(node_sample_set, n_nodes_in_each_round)

            w_accumulate = None
            accumulated = 0

            for n in node_subset:
                model.assign_weight(w_global)
                model.model.train()
#                 print('hahahah')
                for i in range(0, iters_per_round):
                    if use_full_batch:
                        images, labels = sample_full_batch(n)
                    else:
                        images, labels = sample_minibatch(n)

                    images, labels = images.to(device), labels.to(device)

                    if transform_train is not None:
                        images = transform_train(images)

                    model.optimizer.zero_grad()
                    output = model.model(images)
                    loss = model.loss_fn(output, labels)
                    loss.backward()
                    model.optimizer.step()
                w_tmp = model.get_weight()  # deepcopy is already included here
                images_test, labels_test = sample_full_batch(n)
#                 _ ,loss_lo = model.accuracy_im(images_test, labels_test, w_tmp, device)
#                 print(loss_lo)
                w_tmp -= w_global

                if accumulated == 0:  # accumulated weights
                    w_accumulate = w_tmp
                    # Note: w_tmp cannot be used after this
                else:
                    w_accumulate += w_tmp

                accumulated += 1
            worker_subset = [n for n in range(0, n_workers)]
#             if accumulated > 0:
#                 w_global += torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)
            _ , loss_gl = model.accuracy(data_train_loader, w_global, device)
            if n_workers is None and accumulated > 0 or num_iter <= iters_warmup:
                 w_global += torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)
            elif num_iter > iters_warmup:
                for w in worker_subset:
                    model.assign_weight(w_global)
                    model.model.train()

                    for i in range(0, iters_per_round):
                        if use_full_batch:
                            images, labels = sample_full_batch_w(w)
                        else:
                            images, labels = sample_minibatch_w(w)


                        images, labels = images.to(device), labels.to(device)

                        if transform_train is not None:
                            images = transform_train(images)

                        model.optimizer.zero_grad()
                        output = model.model(images)
                        loss = model.loss_fn(output, labels)
                        loss.backward()
                        model.optimizer.step()

                    w_tmp = model.get_weight()  # deepcopy is already included here
                    images_test, labels_test = sample_minibatch_w(w)
                    _ ,loss_lo = model.accuracy_im(images_test, labels_test, w_tmp, device)
                    w_tmp -= w_global
#                     print(loss_lo,loss_gl)
                    # Check if the worker's gradient should be included
                    if include_worker_gradient(w, w_tmp, loss_lo, loss_gl, 0.4, method):
                        print("yes")
                        if accumulated == 0:  # accumulated weights
                            w_accumulate = w_tmp
#                             print('hi-no')
                        else:
                            w_accumulate += w_tmp
#                             print('hi-yes')
                        accumulated += 1
                        
                if accumulated > 0:
                    w_global += torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)

#             if wait_for_all and not warming_up:
#                 num_iter = num_iter + (active_rounds + inactive_rounds) * iters_per_round
#             else:
            num_iter = num_iter + iters_per_round

            if num_iter - last_save_latest >= (active_rounds) * iters_per_round:
                print('Saving model')
                torch.save(model.model.state_dict(), save_model_file)
                last_save_latest = num_iter

#             if num_iter - last_amplify >= p_value * iters_per_round:
#                 if use_global_update and not warming_up:
#                     print('Amplifying update')
#                     w_global += (step_size_global - 1.0) * (w_global - w_global_prev_for_add_update.to(device))
#                     w_global_prev_for_add_update = copy.deepcopy(w_global.to('cpu'))

#                 last_amplify = num_iter

            if save_checkpoint and num_iter - last_save_checkpoint >= iters_checkpoint:
                torch.save(model.model.state_dict(), save_model_file + '-checkpoint-sim-' + str(seed) + '-iter-' + str(num_iter))
                last_save_checkpoint = num_iter

            if num_iter - last_output >= min_iters_per_eval:
                stat.collect_stat(seed, num_iter, model, data_train_loader, data_test_loader, w_global)
                last_output = num_iter

            if num_iter >= max_iter:
                break

#             if periodic_availability:
#                 # Recompute node connectivity
#                 for n in range(n_nodes):
#                     time_until_switch[n] -= 1
#                     if time_until_switch[n] <= 0:
#                         if active_nodes[n] == True:
#                             active_nodes[n] = False
#                             time_until_switch[n] = inactive_rounds
#                         else:
#                             active_nodes[n] = True
#                             time_until_switch[n] = active_rounds

        del model
        del w_global
        del w_accumulate
        torch.cuda.empty_cache()

