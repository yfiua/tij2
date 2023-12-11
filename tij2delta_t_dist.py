#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys

# Save data
def save_dist(filename, delta_t_history):
    bins = np.unique(delta_t_history)
    hist, _ = np.histogram(delta_t_history, np.append(bins, max(bins)+1))

    df_dat = pd.DataFrame({'delta_t': bins, 'freq': hist}).astype({'delta_t': 'int', 'freq': 'int'})
    df_dat.to_csv(filename, sep=' ', header=False, index=False)

    return

# Main
def main(input_file, output_file_delta_t, output_file_delta_t_dist):
    # load data
    df_tij = pd.read_csv(input_file, sep=' |\t', header=None, engine='python')
    df_tij.columns = ['t', 'i', 'j']

    # compute contact duration distribution
    min_t, max_t = df_tij['t'].min(), df_tij['t'].max()
    max_id = np.max(df_tij[['i', 'j']].values) + 1

    ## edge age matrix
    delta_t = np.zeros([max_id, max_id])
    delta_t_history = []

    t_interval = 20
    for t in np.arange(min_t, max_t, t_interval):
        df_cursor = df_tij[np.logical_and(df_tij['t'] >= t, df_tij['t'] < (t + t_interval))]

        # overall
        A = np.zeros([max_id, max_id])
        A[df_cursor['i'].values, df_cursor['j'].values] = 1

        ## edges to disappear
        edge_disappear = np.logical_and(delta_t != 0, A == 0)
        ## record ages, and set ages as 0, for the edges to disappear
        delta_t_history = np.append(delta_t_history, delta_t[edge_disappear])
        delta_t[edge_disappear] = 0
        ## update edge age
        delta_t += A

    # save data
    np.savetxt(output_file_delta_t, delta_t_history, fmt='%i')
    save_dist(output_file_delta_t_dist, delta_t_history)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
