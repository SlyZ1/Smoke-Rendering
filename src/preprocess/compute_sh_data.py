import numpy as np
from scipy.special import legendre
from sympy.physics.wigner import gaunt
import matplotlib.pyplot as plt

# ----------- ZH COEFFS
def unit_rbf(x):
    if not np.isscalar(x):
        x = np.linalg.norm(x)
    return np.exp(-x**2)

def optical_depth(beta=None, b2=None):
    # intergrale  exp(-(t² + b²)) dt
    # = exp(-b²) integrale exp(-t²) <---- gaussian integral
    # = sqrt(pi) exp(-b²)
    # b = sin(beta)
    if beta is None and b2 is None:
        raise ValueError("Both beta and b2 are None.")
    if beta is not None and b2 is not None:
        raise ValueError("Only beta or b2 should be entered.")
    if beta is not None:
        b2 = np.sin(beta)**2
    return np.sqrt(np.pi) * np.exp(-b2)

def compute_zh_coeffs(order=4, n_samples=512):
    mu = np.linspace(-1, 1, n_samples)
    T = optical_depth(b2=(1 - mu**2))
    coeffs = []
    for l in range(order):
        if l % 2 == 1:
            coeffs.append(0)
            continue
        P_l = legendre(l)(mu)
        integrand = T * P_l
        c_l = (2*l + 1) / 2 * np.trapezoid(integrand, mu)
        coeffs.append(c_l)
    return np.array(coeffs)

def ZH_optical_depth(beta, order=4):
    mu = np.cos(beta)
    result = 0.0
    for l in range(order):
        result += coeffs[l] * legendre(l)(mu)
    return result

def lm_to_idx(l, m):
    return l*l + l + m

# -------------- GAUNT TRIPLE PROD

def compute_gaunt_tensor(order=4):
    n = order**2
    tensor = np.zeros((n, n, n))
    for l1 in range(order):
        for m1 in range(-l1, l1+1):
            for l2 in range(order):
                for m2 in range(-l2, l2+1):
                    for l3 in range(order):
                        for m3 in range(-l3, l3+1):
                            if (l1 + l2 + l3) % 2 != 0:
                                continue
                            if (m1 + m2 + m3) > 0:
                                continue
                            if l3 < abs(l1-l2) or l3 > l1+l2:
                                continue

                            i = lm_to_idx(l1, m1)
                            j = lm_to_idx(l2, m2)
                            k = lm_to_idx(l3, m3)
                            tensor[i,j,k] = float(gaunt(l1, l2, l3, m1, m2, m3))
    return tensor

def generate_triple_product_glsl(tensor, order=4):
    n = order**2
    lines = []
    lines.append("void sh_triple_product3(float f[16], float g[16], out float result[16]){\n")
    equals = True
    for i in range(n):
        equals = True
        for j in range(n):
            for k in range(n):
                v = tensor[i,j,k]
                if abs(v) > 1e-10:
                    lines.append(f"\tresult[{i}] {'' if equals else '+'}= {v:.6f} * f[{j}] * g[{k}];\n")
                    equals = False
    lines.append("}")
    with open("sh_triple_product.glsl", "w") as f:
        f.writelines(lines)

# ------------------ EXP *
def compute_a_b_tables(max_magnitude=50, n_table=256, n_samples=1024, plot_res=False):
    t = np.linspace(0, 1, n_table)
    magnitudes = np.sqrt(t) * max_magnitude
    mu = np.linspace(-1, 1, n_samples)
    a_table = np.zeros(n_table)
    b_table = np.zeros(n_table)
    for i, mag in enumerate(magnitudes):
        f_vals = mag * mu
        exp_f = np.exp(f_vals)
        c0 = 0.5 * np.trapezoid(exp_f, mu)
        c1 = 1.5 * np.trapezoid(exp_f * mu, mu)

        # exp_star(f) ~= 1.a + f.b + f².c ...

        a = c0 / np.sqrt(4 * np.pi)
        b = c1 / mag if mag > 1e-8 else 1.0/3.0
        a_table[i] = a
        b_table[i] = b
    if plot_res:
        plt.plot(magnitudes, a_table, label='a')
        plt.plot(magnitudes, b_table, label='b')
        plt.show()
    exp_data = [max_magnitude] + a_table + b_table
    exp_data.tofile("exp_data.bin")

# ------------- SH CONV
def generate_sh_conv_glsl(order=4):
    lines = []
    lines.append("void sh_conv3(float f[16], float g[16], out float result[16]){\n")
    equals = np.ones(order**2)
    for l in range(order):
        for m in range(-l, l+1):
            factor = np.sqrt(4 * np.pi / (2 * l + 1))
            i = lm_to_idx(l, m)
            j = lm_to_idx(l, 0)
            equal = equals[i]
            lines.append(f"\tresult[{i}] {'' if equal else '+'}= {factor:.6f} * f[{i}] * g[{j}];\n")
            equals[i] = 0
    lines.append("}")
    with open("sh_conv.glsl", "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    order = 4

    # ZH COEFFS
    coeffs = compute_zh_coeffs(order=order, n_samples=10000)
    coeffs.tofile("zh.bin")
    print(f"Zonal coefficients for order {order}: \n{coeffs}")
    # betas = np.linspace(0, np.pi, 6)
    # print(ZH_optical_depth(betas))
    # print(optical_depth(betas))

    # GAUNT TENSOR
    gamma = compute_gaunt_tensor()
    # generate_triple_product_glsl(gamma)
    print("Gamma written in glsl file.")

    # EXP *
    max_density_mag = 1.85
    sigma_t = 20
    max_mag = np.min([max_density_mag * sigma_t, 15])
    compute_a_b_tables(n_table=256, max_magnitude=max_mag, plot_res=False)
    print("Exp* parameters computed.")

    # SH CONV
    generate_sh_conv_glsl()
    print("SH conv written in glsl file.")