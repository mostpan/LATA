def correct(n_node, edges, out_loss, x_label,cor_fraction):

    l_list = []
    for u, v in edges:
        if u < v:
            luv = (out_loss[u * n_node + v] + out_loss[v * n_node + u]) / 2
            l_list.append((u, v, luv))
    l_list.sort(key=lambda x: abs(x[2]), reverse=True)

    sa_len = int(len(l_list) * cor_fraction)

    for si in range(sa_len):
        u, v, luv = l_list[si]
        x_label[u * n_node + v], x_label[v * n_node + u] = 0, 0

    return x_label