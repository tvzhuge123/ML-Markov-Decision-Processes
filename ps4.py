import numpy as np
import pandas as pd
import util
import blackjack
import mdptoolbox, mdptoolbox.example


algos = ["Policy Iteration", "Value Iteration"]
ql_decays = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]


def process(prob='Forest Management'):
    if prob == 'Forest Management':
        n_states = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    else:
        n_states = [723]

    df_iter = pd.DataFrame(columns=algos)
    df_time = pd.DataFrame(columns=algos)
    df_v = pd.DataFrame(columns=algos)
    for n_state in n_states:
        print(n_state)

        if prob == 'Forest Management':
            P, R = mdptoolbox.example.forest(S=n_state, r1=n_state + 1, r2=2, p=0.1, is_sparse=False)
        else:
            bj = blackjack.BlackJack()
            P, R = bj.get_matrices()

        df_iter_tmp = pd.DataFrame(columns=algos)
        df_time_tmp = pd.DataFrame(columns=algos)
        df_v_tmp = pd.DataFrame(columns=algos)
        for i in range(10):
            # run
            pi, vi = util.run(P, R, 0.9)
            # stats
            df_iter_tmp.loc[len(df_iter_tmp)] = [pi.iter, vi.iter]
            df_time_tmp.loc[len(df_time_tmp)] = [pi.time, vi.time]
            df_v_tmp.loc[len(df_time_tmp)] = [pi.V[0], vi.V[0]]
            if pi.policy != vi.policy:
                print(n_state, pi.policy, vi.policy)

        df_iter.loc[len(df_iter)] = df_iter_tmp.mean(axis=0)
        df_time.loc[len(df_time)] = df_time_tmp.mean(axis=0)
        df_v.loc[len(df_time)] = df_v_tmp.mean(axis=0)

    # plot
    df_iter.set_index(pd.Index(n_states), inplace=True)
    df_time.set_index(pd.Index(n_states), inplace=True)
    df_v.set_index(pd.Index(n_states), inplace=True)
    util.plot(df_iter, df_time, df_v)



    df_iter = pd.DataFrame(columns=algos)
    df_time = pd.DataFrame(columns=algos)
    df_v = pd.DataFrame(columns=algos)
    for decay in ql_decays:
        if prob == 'Forest Management':
            P, R = mdptoolbox.example.forest(S=100, r1=100 + 1, r2=2, p=0.1, is_sparse=False)
        else:
            bj = blackjack.BlackJack()
            P, R = bj.get_matrices()

        df_iter_tmp = pd.DataFrame(columns=algos)
        df_time_tmp = pd.DataFrame(columns=algos)
        df_v_tmp = pd.DataFrame(columns=algos)
        for i in range(10):
            # run
            pi, vi = util.run(P, R, decay)
            # stats
            df_iter_tmp.loc[len(df_iter_tmp)] = [pi.iter, vi.iter]
            df_time_tmp.loc[len(df_time_tmp)] = [pi.time, vi.time]
            df_v_tmp.loc[len(df_time_tmp)] = [pi.V[0], vi.V[0]]
            if pi.policy != vi.policy:
                print(100, pi.policy, vi.policy)

        df_iter.loc[len(df_iter)] = df_iter_tmp.mean(axis=0)
        df_time.loc[len(df_time)] = df_time_tmp.mean(axis=0)
        df_v.loc[len(df_time)] = df_v_tmp.mean(axis=0)

    # plot
    df_iter.set_index(pd.Index(ql_decays), inplace=True)
    df_time.set_index(pd.Index(ql_decays), inplace=True)
    df_v.set_index(pd.Index(ql_decays), inplace=True)
    util.plot(df_iter, df_time, df_v, xlabel='Discount Factor')



    # Q learning
    df_time = pd.DataFrame(columns=ql_decays)
    df_v = pd.DataFrame(columns=ql_decays)
    for n_state in n_states:
        print(n_state)
        P, R = mdptoolbox.example.forest(S=n_state, r1=n_state + 1, r2=2, p=0.1, is_sparse=False)
        v_list, time_list = [], []
        for decay in ql_decays:
            v_list_tmp, time_list_tmp = [], []
            for i in range(10):
                v, time = util.run_ql(P, R, decay)
                v_list_tmp.append(v)
                time_list_tmp.append(time)
            v_list.append(np.max(v_list_tmp))
            time_list.append(np.mean(time_list_tmp))
        df_time.loc[len(df_time)] = time_list
        df_v.loc[len(df_time)] = v_list

    # plot
    df_time.set_index(pd.Index(n_states), inplace=True)
    df_v.set_index(pd.Index(n_states), inplace=True)
    util.plot(None, df_time=df_time, df_v=df_v)

    return


def main():
    process('Forest Management')
    process('BlackJack')
    return


if __name__ == '__main__':
    main()
