import random
import numpy as np

def get_E_A(x, i, j):
    du, dv = x[i].sum(), x[j].sum()
    puv = min(du, dv) / max(du, dv)
    return puv


def get_E_B(x, i, j):
    return (x[i] * x[j]).sum() / len(x)


def get_E_distr(x, get_E_X):
    bins = 20
    p_list = []
    no_zids = x.nonzero()
    no_zids = [(no_zids[0][i], no_zids[1][i]) for i in range(len(no_zids[0]))]
    for u, v in no_zids:
        if u < v:
            puv = get_E_X(x, u, v)
            p_list.append(puv)
    x_s, x_e = np.min(p_list), np.max(p_list)
    x_w = (x_e - x_s) / bins

    max_cnt, distr = 0, []
    for i in range(bins):
        cnt = 0
        for j in range(len(p_list)):
            if x_s + x_w * i <= p_list[j] and p_list[j] < x_s + x_w * (i + 1):
                cnt += 1
        distr.append([x_s + x_w * i, x_s + x_w * (i + 1), cnt])
        max_cnt = max(max_cnt, cnt)
    max_cnt = max_cnt
    for i in range(bins):
        distr[i][2] /= max_cnt
    return distr


def augment(x, distr_a, distr_b, aug_fraction):
    x1 = x.copy()
    x2 = x.copy()

    n_node = len(x)
    n_add = int(x.sum() / 2 * aug_fraction)
    for i in range(n_add):
        while True:
            u, v = random.randint(0, n_node - 1), random.randint(0, n_node - 1)
            if u == v or x1[u, v] == 1:
                continue
            x_k = get_E_A(x, u, v)
            puv = -1
            for x_s, x_e, x_v in distr_a:
                if x_s <= x_k and x_k < x_e:
                    puv = x_v
                    break

            p = random.random()
            if p < puv:
                x1[u, v], x1[v, u] = 1, 1
                break

    for i in range(n_add):
        while True:
            u, v = random.randint(0, n_node - 1), random.randint(0, n_node - 1)
            if u == v or x2[u, v] == 1:
                continue
            x_k = get_E_B(x, u, v)
            puv = -1
            for x_s, x_e, x_v in distr_b:
                if x_s <= x_k and x_k < x_e:
                    puv = x_v
                    break

            p = random.random()
            if p < puv:
                x2[u, v], x2[v, u] = 1, 1
                break

    return x1, x2