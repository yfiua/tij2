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
    df_tij = pd.read_csv(f_input, sep='\t', header=None)
    df_tij.columns = ['t', 'i', 'j']

    # Map node IDs to 0, 1, 2, ...
    node_ids = np.sort(np.unique(df_tij[['i', 'j']].values))
    node_id_mapping = { node_id : i for i, node_id in enumerate(node_ids) }
    df_tij[['i', 'j']] = df_tij[['i', 'j']].replace(node_id_mapping)

    # make it undirected
    df_tij = pd.concat([df_tij, df_tij.rename(columns={'i': 'j', 'j': 'i'})], ignore_index=True)

    # Compute contact duration distribution
    min_t, max_t = df_tij['t'].min(), df_tij['t'].max()
    max_id = df_tij['j'].max() + 1

    ## triangle age matrix
    delta_t = {}
    delta_t_history, i_history, j_history, k_history = [], [], [], []

    ## function to get triangles, assuming i < j < k
    def get_triangles(A):
        B = A @ A
        B = np.logical_and(A, B)
        i, j = np.where(B)

        for i_, j_ in zip(i, j):
            if i_ < j_:
                k = np.where(np.logical_and(A[i_, :], A[j_, :]))[0]
                for k_ in k:
                    if j_ < k_:
                        yield i_, j_, k_

    t_interval = 20
    for t in tqdm(np.arange(min_t, max_t, t_interval)):
        df_cursor = df_tij[np.logical_and(df_tij['t'] >= t, df_tij['t'] < (t + t_interval))]

        # overall
        A = np.zeros([max_id, max_id])
        A[df_cursor['i'].values, df_cursor['j'].values] = 1

        # set for the triangles
        A_3 = {(i, j, k) for i, j, k in get_triangles(A)}

        ## triangles to disappear when delta_t > 0 and A_3 == 0
        triangles_disappear = {k: delta_t[k] for k in delta_t.keys() if k not in A_3}

        ## get i and j for the edges to disappear
        if len(triangles_disappear) > 0:
            i_disappear, j_disappear, k_disappear = zip(*triangles_disappear.keys())

            ## record ages, i, j, and set ages as 0, for the edges to disappear
            delta_t_history = np.append(delta_t_history, list(triangles_disappear.values()))
            i_history = np.append(i_history, i_disappear)
            j_history = np.append(j_history, j_disappear)
            k_history = np.append(k_history, k_disappear)

            # remove the triangles to disappear
            for k in triangles_disappear.keys():
                del delta_t[k]

        ## update triangle age
        for k in A_3:
            if k in delta_t:
                delta_t[k] += 1
            else:
                delta_t[k] = 1

    # Save data as CSV
    df_triangles = pd.DataFrame({'i': i_history, 'j': j_history, 'k': k_history, 'delta_t': delta_t_history}).astype(int)
    df_triangles.to_csv(f_output, index=False)

if __name__ == '__main__':
    f_input = sys.argv[1]
    f_output = sys.argv[2]

    main(f_input, f_output)
