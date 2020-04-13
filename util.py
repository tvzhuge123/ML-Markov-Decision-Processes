import time
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt



def run(P, R, d, max_iter=100000, epsilon=0.01):
    # Policy Iteration
    pi = mdptoolbox.mdp.PolicyIterationModified(transitions=P, reward=R, discount=d,
                                                max_iter=max_iter, epsilon=epsilon)
    pi.run()

    # Value Iteration
    vi = mdptoolbox.mdp.ValueIteration(transitions=P, reward=R, discount=d,
                                       max_iter=max_iter, epsilon=epsilon, initial_value=0)
    vi.run()

    return(pi, vi)


def run_ql(P, R, d, max_iter=10000):
    ql = mdptoolbox.mdp.QLearning(transitions=P, reward=R, discount=d, n_iter=max_iter)
    tic = time.process_time()
    ql.run()
    toc = time.process_time()
    ql_time = toc - tic

    return(ql.V[0], ql_time)


def plot(df_iter=None, df_time=None, df_v=None, xlabel='Number of States'):
    if df_iter is not None:
        fig = plt.figure(figsize=(10, 10))

        ax_iter = fig.add_subplot(3, 1, 1)
        ax_iter.set_ylabel("Number of Iterations")

        ax_time = fig.add_subplot(3, 1, 2)
        ax_time.set_ylabel("CPU Time (s)")

        ax_v = fig.add_subplot(3, 1, 3)
        ax_v.set_ylabel("Value at first state")
        ax_v.set_xlabel(xlabel)

        df_iter.plot(grid=True, marker='o', xticks=df_iter.index, ax=ax_iter)
        df_time.plot(grid=True, marker='o', xticks=df_time.index, ax=ax_time)
        df_v.plot(grid=True, marker='o', xticks=df_v.index, ax=ax_v)
    else:
        fig = plt.figure(figsize=(10, 10))

        ax_time = fig.add_subplot(2, 1, 1)
        ax_time.set_ylabel("CPU Time (s)")

        ax_v = fig.add_subplot(2, 1, 2)
        ax_v.set_ylabel("Value at first state")
        ax_v.set_xlabel(xlabel)

        df_time.plot(grid=True, marker='o', xticks=df_time.index, ax=ax_time)
        df_v.plot(grid=True, marker='o', xticks=df_v.index, ax=ax_v)

    plt.show()
