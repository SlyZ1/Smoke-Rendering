import numpy as np
import torch
from tqdm import tqdm
from copy import copy
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")

def load_density(path):
    data = np.load(path)

    D = data[data.files[0]].squeeze(-1)
    D = np.transpose(D, (2, 1, 0))
    D = D.astype(np.float32)
    counts, bins = np.histogram(D.flatten(), bins=200)
    D /= D.max()
    n_clamp = 100
    threshold = np.partition(D.flatten(), -n_clamp)[-n_clamp]
    D = np.clip(D, 0, threshold)
    D /= D.max()
    D.tofile("density.bin")

    D_t = copy(D)
    X = np.indices(D.shape)
    X = X.reshape(3, -1).T
    distances = np.linalg.norm(X.reshape((D.shape[0], D.shape[1], D.shape[2], 3)) / np.array(D.shape) - [0.5,0.5,0.5], axis=-1)
    distances /= distances.max()
    #D_t = D_t + 0.01 * distances
    X = X.astype(np.float32) / np.array(D.shape)

    D_shape = D.shape

    print("density informations:")
    print("\t-shape:", D.shape)
    print("\t-type:", D.dtype)
    print("\t-min-max:", D.min(), D.max())
    print("\t-mean:", np.mean(D))
    counts, bins = np.histogram(D.flatten(), bins=200)
    plt.bar(bins[:-1], counts, width=np.diff(bins))
    plt.yscale('log')  # log scale pour voir les petites valeurs
    plt.show()

    X = torch.from_numpy(X).float().to(device)
    D_t = torch.from_numpy(D_t.flatten()).float().to(device)
    D = torch.from_numpy(D.flatten()).float().to(device)

    return D_t, D_shape, X, D

def load_params(load_from_data, numRBF):
    if load_from_data:
        params = np.load("raw_params.npz")
        C = torch.nn.Parameter(torch.from_numpy(params['C']).to(device))
        R = torch.nn.Parameter(torch.from_numpy(params['R']).to(device))
        W = torch.nn.Parameter(torch.from_numpy(params['W']).to(device))
    else:
        mask = D >= D.mean()
        X_masked = X[mask]
        D_masked = D[mask]
        indices = torch.linspace(0, X_masked.shape[0] - 1, numRBF).long()
        C = torch.nn.Parameter(X_masked[indices].float().to(device))
        R = torch.nn.Parameter((torch.rand(numRBF) * 0.05).to(device))
        W = torch.nn.Parameter(D_masked[indices].to(device))
    C = C.to(device)
    R = R.to(device)
    W = W.to(device)
    return C, R, W

def compute_WdotB(X_, C_, R_, W_):
    dists = torch.cdist(X_, C_)
    R_abs = R_.abs().clamp(min=1e-6)
    mask = (dists < 3 * R_abs[None, :])
    U = dists[mask] / R_abs.expand_as(dists)[mask]
    B = torch.zeros_like(dists)
    B[mask] = torch.exp(-U**2)
    return (W_[None, :].abs() * B).sum(dim=-1)

def teleport(R_, W_, D_shape, Res):
    error = Res.abs()
    max_error_pos = torch.argmax(error)
    max_error_pos = torch.argwhere(error >= error[max_error_pos] * 0.7)
    idx = torch.randint(0, max_error_pos.shape[0], (1,)).item()
    max_error_pos = max_error_pos[idx][0]
    mex, mey, mez = torch.unravel_index(max_error_pos, D_shape)
    new_pos = torch.stack([mex, mey, mez]) / torch.tensor(D_shape, dtype=torch.float32).to(device)

    rbf_importance = W_.abs() * (R_.abs() ** 3)
    least_important_rbf = torch.argmin(rbf_importance)
    least_important_rbf = torch.argwhere(rbf_importance <= rbf_importance[least_important_rbf])
    idx = torch.randint(0, least_important_rbf.shape[0], (1,)).item()
    least_important_rbf = least_important_rbf[idx]

    print(f"Max error position : ({mex.cpu().numpy()}, {mey.cpu().numpy()}, {mez.cpu().numpy()})")
    print(f"Min importance RBF index : {least_important_rbf.cpu().numpy()}")

    return least_important_rbf, new_pos

def save_data(Res_N, Res_P, WdotB, C_, R_, W_):
    Res_N = Res_N.abs().detach().cpu().numpy()
    Res_N.tofile("Res_N.bin")

    Res_P = Res_P.abs().detach().cpu().numpy()
    Res_P.tofile("Res_P.bin")

    WdotB = WdotB.detach().cpu().numpy()
    WdotB.tofile("density_tilde.bin")

    np.savez("raw_params.npz", 
        C=C_.detach().cpu().numpy(),
        R=R_.detach().cpu().numpy(),
        W=W_.detach().cpu().numpy()
    )

def optimize_rbfs(stepsize, X_, D_):
    for _ in tqdm(range(stepsize)):
        optimizer.zero_grad()
        chunks = []
        for i in range(0, X_.shape[0], batchsize):
            X_b = X_[i:i+batchsize]
            chunks.append(compute_WdotB(X_b, C, R, W))
        WdotB = torch.cat(chunks)
        Res = D_ - WdotB
        loss = torch.sum(Res ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            R.clamp_(0.015, 0.1)
            W.clamp_min_(0.01)
    return WdotB, Res, loss

def update_lrs(optimizer, speed=0.1):
    optimizer.param_groups[0]['lr'] *= 1 - speed
    optimizer.param_groups[0]['lr'] = np.max([optimizer.param_groups[0]['lr'], 0.001])
    optimizer.param_groups[1]['lr'] *= 1 - speed
    optimizer.param_groups[1]['lr'] = np.max([optimizer.param_groups[1]['lr'], 0.001])
    optimizer.param_groups[2]['lr'] *= 1 - speed
    optimizer.param_groups[2]['lr'] = np.max([optimizer.param_groups[2]['lr'], 0.001])
    return optimizer


if __name__ == "__main__":
    D, D_shape, X, D_original = load_density("raw_density.npz")
    C, R, W = load_params(load_from_data=False, numRBF=400)
    batchsize = 4096*16
    optimizer = torch.optim.Adam([
        {'params': [W], 'lr': 0.005},
        {'params': [C], 'lr': 0.005},
        {'params': [R], 'lr': 0.005},
    ])
    stepsize = 10
    for step in range(500):
        WdotB, Res, loss = optimize_rbfs(stepsize=stepsize, X_=X, D_=D)
        pg = optimizer.param_groups
        print(f"D_tilde max: {WdotB.max()}")
        print(f"step {(step + 1) * stepsize}, loss: {loss.item():.1f}, lrs: [{pg[0]['lr']:.3}, {pg[1]['lr']:.3}, {pg[2]['lr']:.3}]")
        #optimizer = update_lrs(optimizer, speed=0.05)
        
        # Teleport
        if (step + 1) % 1 == 0:
            least_important_rbf, new_pos = teleport(R, W, D_shape, Res)
            with torch.no_grad():
                C[least_important_rbf] = new_pos
                R[least_important_rbf] = torch.tensor(0.05, device=device)
                W[least_important_rbf] = 0.9

        Res_N = Res.clone()
        Res_N[Res_N > 0] = 0
        Res_N = Res_N.abs()
        
        Res_P = D_original - WdotB
        Res_P[Res_P < 0] = 0
        save_data(Res_N, Res_P, WdotB, C, R, W)