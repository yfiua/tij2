#!/usr/bin/env python3
# coding: utf-8

import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import sparse

# main
def main(f_input, f_output):
    # Load network
    df_tij = pd.read_csv(f_input, sep=' ', header=None)
    df_tij.columns = ['t', 'i', 'j']

    # Map node IDs to 0, 1, 2, ...
    node_ids = np.sort(np.unique(df_tij[['i', 'j']].values))
    node_id_mapping = { node_id : i for i, node_id in enumerate(node_ids) }
    df_tij[['i', 'j']] = df_tij[['i', 'j']].replace(node_id_mapping)

    # make it undirected
    df_tij = pd.concat([df_tij, df_tij.rename(columns={'i': 'j', 'j': 'i'})], ignore_index=True)

    # get boundaries for t and node IDs
    min_t, max_t = df_tij['t'].min(), df_tij['t'].max()
    max_id = df_tij['j'].max() + 1

    ## tetrahedra age matrix
    delta_t = {}
    delta_t_history, i_history, j_history, k_history, l_history = [], [], [], [], []

    ## function to get triangles, assuming i < j < k
    def get_triangles(A):
        # get triangles (i, j are connected by an edge and a 2-hop path)
        B = np.logical_and(A, A @ A)
        i, j = np.where(B)

        for i_, j_ in zip(i, j):
            if i_ < j_:
                k = np.where(np.logical_and(A[i_, :], A[j_, :]))[0]
                for k_ in k:
                    if j_ < k_:
                        yield i_, j_, k_

    ## function to get tetrahedra
    def get_tetrahedra(A):
        # get triangles first
        i, j, k = zip(*get_triangles(A))

        for i_, j_, k_ in zip(i, j, k):
            # l is the fourth node that forms a tetrahedron with i, j, k
            l = np.where(A[i_, :] * A[j_, :] * A[k_, :])[0]
            for l_ in l:
                if k_ < l_:
                    yield i_, j_, k_, l_

    t_interval = 20
    for t in tqdm(np.arange(min_t, max_t, t_interval)):
        df_cursor = df_tij[np.logical_and(df_tij['t'] >= t, df_tij['t'] < (t + t_interval))]

        # overall
        A = np.zeros([max_id, max_id])
        A[df_cursor['i'].values, df_cursor['j'].values] = 1

        # set for all tetrahedra
        try:
            set_tetrahedra = {(i, j, k, l) for i, j, k, l in get_tetrahedra(A)}
        except ValueError:
            # if no tetrahedra found, continue
            set_tetrahedra = set()

        ## tetrahedra to disappear, i.e., delta_t > 0 and not in set_tetrahedra
        tetrahedra_disappear = {k: delta_t[k] for k in delta_t.keys() if k not in set_tetrahedra}

        ## get indices for the tetrahedra to disappear
        if len(tetrahedra_disappear) > 0:
            i_disappear, j_disappear, k_disappear, l_disappear = zip(*tetrahedra_disappear.keys())

            ## record ages, i, j, k, l, and set ages as 0, for the edges to disappear
            delta_t_history = np.append(delta_t_history, list(tetrahedra_disappear.values()))
            i_history = np.append(i_history, i_disappear)
            j_history = np.append(j_history, j_disappear)
            k_history = np.append(k_history, k_disappear)
            l_history = np.append(l_history, l_disappear)

            # remove the tetrahedra to disappear
            for k in tetrahedra_disappear.keys():
                del delta_t[k]

        ## update tetrahedra age
        for k in set_tetrahedra:
            if k in delta_t:
                delta_t[k] += 1
            else:
                delta_t[k] = 1

    # Save data as CSV
    df_tetrahedra =  pd.DataFrame({'i': i_history, 'j': j_history, 'k': k_history, 'l': l_history, 'delta_t': delta_t_history}).astype(int)
    df_tetrahedra.to_csv(f_output, index=False)

if __name__ == '__main__':
    f_input = sys.argv[1]
    f_output = sys.argv[2]

    main(f_input, f_output)
