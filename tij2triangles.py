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

    ## all triangles
    df_triangles = pd.DataFrame(columns=['t', 'i', 'j', 'k'])

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

    t_interval = 20
    for t in tqdm(np.arange(min_t, max_t, t_interval)):
        df_cursor = df_tij[np.logical_and(df_tij['t'] >= t, df_tij['t'] < (t + t_interval))]

        # overall
        A = np.zeros([max_id, max_id])
        A[df_cursor['i'].values, df_cursor['j'].values] = 1

        # triangles
        try:
            i_, j_, k_ = zip(*get_triangles(A))
            df_triangles = pd.concat([df_triangles, pd.DataFrame({'t': t, 'i': i_, 'j': j_, 'k': k_})], ignore_index=True)
        except ValueError:
            # if no triangles found, continue
            continue

    # Save data as CSV
    df_triangles = df_triangles.astype(int)
    df_triangles.to_csv(f_output, index=False)

if __name__ == '__main__':
    f_input = sys.argv[1]
    f_output = sys.argv[2]

    main(f_input, f_output)
