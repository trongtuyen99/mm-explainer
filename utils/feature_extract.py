import torch
import pickle

def _processinput(inp, input_to_float=False):
    if input_to_float:
        return inp.float()
    else:
        return inp


@torch.no_grad()
def get_features(model, data_loader, use_modal):
    model.eval()
    features = []
    for j in data_loader:
        x = [i.float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
             for i in j[:-1]]
        x = [x[i] for i in use_modal]
        x = [v.permute(0, 2, 1)
             for v in x]
        x = [X[:, :, list(range(5, 25, 1))] for X in x]
        proj_x = [model.fuse.proj[i](x[i]) for i in range(2)]
        features.append(proj_x)
    return features

def read_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def write_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)