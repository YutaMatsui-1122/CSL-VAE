import numpy as np
def multi_label_metrics(T_lengths, Y_lengths, Y_and_T_lengths, Y_or_T_lengths):
    #refer to https://qiita.com/jyori112/items/110596b4f04e4e1a3c9b
    #         https://www.researchgate.net/publication/266888594_A_Literature_Survey_on_Algorithms_for_Multi-label_Learning
    n = len(T_lengths)
    accuracy  = 1 / n * np.sum(Y_and_T_lengths / Y_or_T_lengths)
    recall    = 1 / n * np.sum(Y_and_T_lengths / T_lengths)
    precision = 1 / n * np.sum(Y_and_T_lengths / Y_lengths)
    return accuracy, recall, precision

T_lengths = np.array([5,5,5,5,5])
Y_lengths = np.array([5,5,5,5,5])
Y_and_T_lengths = np.array([4,4,4,4,4])
Y_or_T_lengths = np.array([5,5,5,5,5])

print(multi_label_metrics(T_lengths, Y_lengths, Y_and_T_lengths, Y_or_T_lengths))