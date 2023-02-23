import time
import math

def compute_probability(A, B, Pi, O, T):
    try:
        alpha_matrix, c_matrix = compute_alpha_matrix(A, B, Pi, O, T)
        N = len(A)
        probability = 0
        for i in range(N):
            probability += alpha_matrix[T - 1][i]

        return probability
    except:
        return 10000


def compute_prob_obs(A, B, pi, obs, T):
    try:
        alpha_matrix, c_matrix = compute_alpha_matrix(A, B, pi, obs, T)
        prob = 0
        for i in range(len(c_matrix)):
            prob -= math.log10(c_matrix[i])

        return prob
    except ZeroDivisionError:
        return -1000000


def compute_beta_matrix(A, B, Pi, O, T, c_matrix):
    rows = T  # T (number of Observations)
    cols = len(A)  # N (number of states)
    beta_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    # fill in the initial states
    for j in range(cols):
        beta_matrix[T - 1][j] = c_matrix[T - 1]

    # fill the other states
    for t in range(T - 2, 0, -1):
        for i in range(cols):
            for j in range(cols):
                beta_matrix[t][i] += beta_matrix[t + 1][j] * B[j][O[t + 1]] * A[i][j]
            beta_matrix[t][i] = beta_matrix[t][i] * c_matrix[t]
    return beta_matrix


def compute_alpha_matrix(A, B, Pi, O, T):
    rows = T  # T (number of Observations)
    cols = len(A)  # N (number of states)

    alpha_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    c_matrix = [0 for _ in range(T)]

    # rescaling factor
    c_0 = 0

    # fill in the initial states
    for j in range(cols):
        pi_debug = Pi[0][j]
        b_debug = B[j][O[0]]
        #alpha_matrix[0][j] = Pi[0][j] * B[j][O[0]]
        alpha_matrix[0][j] = pi_debug * b_debug
        c_0 += alpha_matrix[0][j]

    # rescale
    c_0 = 1 / c_0
    c_matrix[0] = c_0
    for j in range(cols):
        alpha_matrix[0][j] = c_0 * alpha_matrix[0][j]

    # fill the other states
    for t in range(1, rows):
        c_t = 0
        for i in range(cols):
            for j in range(cols):
                alpha_matrix[t][i] += alpha_matrix[t - 1][j] * A[j][i]
            alpha_matrix[t][i] *= B[i][O[t]]
            c_t += alpha_matrix[t][i]

        # rescale
        c_t = 1 / c_t
        c_matrix[t] = c_t
        for i in range(cols):
            alpha_matrix[t][i] = c_t * alpha_matrix[t][i]

    return alpha_matrix, c_matrix


def compute_di_gamma_function(t, i, j, A, B, O, alpha_matrix, beta_matrix, T):
    numerator = alpha_matrix[t][i] * A[i][j] * B[j][O[t + 1]] * beta_matrix[t + 1][j]
    denomenator = 0
    for k in range(len(A)):
        denomenator += alpha_matrix[T - 1][k]
    return numerator / denomenator


def compute_gamma_function(t, i, A, B, O, alpha_matrix, beta_matrix):
    result = 0
    for j in range(len(A)):
        result += compute_di_gamma_function(t, i, j, A, B, O, alpha_matrix, beta_matrix)

    return result


# ----------------------------------------------------------------------------------------------------------------------
def compute_di_gamma_matrix(A, B, O, alpha_matrix, beta_matrix, T):
    N = len(A)

    # T*N*N
    di_gamma_matrix = [[[0] * N for n in range(N)] for t in range(T)]

    denom = 0
    for k in range(N):
        denom += alpha_matrix[T - 1][k]

    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                num = alpha_matrix[t][i] * A[i][j] * B[j][O[t + 1]] * beta_matrix[t + 1][j]
                di_gamma_matrix[t][i][j] = num / denom

    return di_gamma_matrix


def compute_gamma_matrix(A, B, O, alpha_matrix, beta_matrix, T):
    N = len(A)

    # T*N
    gamma_matrix = [[0 for _ in range(N)] for _ in range(T)]
    di_gamma_matrix = compute_di_gamma_matrix(A, B, O, alpha_matrix, beta_matrix, T)

    for t in range(T):
        for i in range(N):
            for j in range(N):
                gamma_matrix[t][i] += di_gamma_matrix[t][i][j]

    #    for i in range(N):
    #        gamma_matrix[T - 1][i] = alpha_matrix[T - 1][i]

    return gamma_matrix


# -----------------------------------------------------------------------------------------------------------------------

def Baum_Welch_algorithm(A, B, Pi, O, T, iter, start_time):
    N = len(A)
    K = len(B[0])

    # 2. Compute alpha, beta matrices
    alpha_matrix, c_matrix = compute_alpha_matrix(A, B, Pi, O, T)
    beta_matrix = compute_beta_matrix(A, B, Pi, O, T, c_matrix)

    # 3. Re-estimate the model
    new_A = [[0 for _ in range(N)] for _ in range(N)]
    # N*K
    new_B = [[0 for _ in range(K)] for _ in range(N)]
    new_Pi = [[0 for _ in range(N)]]

    di_gamma_matrix = compute_di_gamma_matrix(A, B, O, alpha_matrix, beta_matrix, T)
    gamma_matrix = compute_gamma_matrix(A, B, O, alpha_matrix, beta_matrix, T)

    # re-estimate A
    for i in range(N):
        for j in range(N):
            denominator = 0
            for t in range(T - 1):
                denominator += gamma_matrix[t][i]

            numerator = 0
            for t in range(T - 1):
                numerator += di_gamma_matrix[t][i][j]
            new_A[i][j] = numerator / denominator

    # re-estimate B
    for j in range(N):
        for k in range(K):
            denominator = 0
            for t in range(T - 1):
                denominator += gamma_matrix[t][j]

            numerator = 0
            for t in range(T - 1):
                if O[t] == k:
                    numerator += gamma_matrix[t][j]
            new_B[j][k] = numerator / denominator

    # re-estimate Pi
    for i in range(N):
        new_Pi[0][i] = gamma_matrix[0][i]

    iter += 1

    if iter < 30 and time.time() < start_time + 5 - 1:
        return Baum_Welch_algorithm(new_A, new_B, new_Pi, O, T, iter, start_time)
    else:
        return new_A, new_B, new_Pi


