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

    ## all K5s
    df_k5s = pd.DataFrame(columns=['t', 'i', 'j', 'k', 'l', 'm'])

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

    # function to get K5s
    def get_k5s(A):
        # get tetrahedra first
        i, j, k, l = zip(*get_tetrahedra(A))

        for i_, j_, k_, l_ in zip(i, j, k, l):
            # m is the fifth node that forms a K5 with i, j, k, l
            m = np.where(A[i_, :] * A[j_, :] * A[k_, :] * A[l_, :])[0]
            for m_ in m:
                if l_ < m_:
                    yield i_, j_, k_, l_, m_

    t_interval = 20
    for t in tqdm(np.arange(min_t, max_t, t_interval)):
        df_cursor = df_tij[np.logical_and(df_tij['t'] >= t, df_tij['t'] < (t + t_interval))]

        # overall
        A = np.zeros([max_id, max_id])
        A[df_cursor['i'].values, df_cursor['j'].values] = 1

        # K5s
        try:
            i_, j_, k_, l_, m_ = zip(*get_k5s(A))
            df_k5s = pd.concat([df_k5s, pd.DataFrame({'t': t, 'i': i_, 'j': j_, 'k': k_, 'l': l_, 'm': m_})], ignore_index=True)
        except ValueError:
            # if no K5 found, continue
            continue

    # Save data as CSV
    df_k5s = df_k5s.astype(int)
    df_k5s.to_csv(f_output, index=False)

if __name__ == '__main__':
    f_input = sys.argv[1]
    f_output = sys.argv[2]

    main(f_input, f_output)
