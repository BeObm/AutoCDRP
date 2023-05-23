# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:48:36 2020

@author: TOP Artes
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
# from Orange.evaluation import compute_CD, graph_ranks

df_results = pd.read_excel('data/comparison_result_autocdrp.xlsx', index_col=0,header=0)
results_array = df_results.values

wilcoxon(
    results_array[2],
    results_array[4],
    zero_method='zsplit')

"""
The Wilcoxon signed-rank test outputs a p-value = 0.13801073756865956. If we consider a significance level (α) of 0.05 we can conclude that Random Forest and
Neural Network performances are not equivalent.
Comparing multiple classifiers over multiple datasets
The Wilcoxon signed-rank test was not designed to compare multiple random variables. So, when comparing multiple classifiers, an
"intuitive" approach would be to apply the Wilcoxon test to all possible pairs. However, when multiple tests are conducted, some of them will
reject the null hypothesis only by chance (Demšar, 2006).
For the comparison of multiple classifiers, Demšar (2006) recommends the Friedman test"""

friedmanchisquare(*results_array)
"""
pvalue = 0.009533014205069796
The Friedman test outputs a very small p-value. For many significance levels (α) we can conclude that the performances of all algorithms are
not equivalent.
Considering that the null-hypothesis was rejected, we usually have two scenarios for a post-hoc test (Demšar, 2006):
All classifiers are compared to each other. In this case we apply the Nemenyi post-hoc test.
All classifiers are compared to a control classifier. In this scenario we apply the Bonferroni-Dunn post-hoc test."""

algorithms = list(df_results.columns)
"""
Calculating the ranks of the algorithms for each dataset. The value of p is multipled by -1
# because the rankdata method ranks from the smallest to the greatest performance values.
# Since we are considering AUC as our performance measure, we want larger values to be best ranked."""


ranks = np.array([rankdata(-p) for p in results_array])
# Calculating the average ranks.
average_ranks = np.mean(ranks, axis=0)
print('\n'.join('{} average rank: {}'.format(a, r) for a, r in zip(algorithms, average_ranks)))

names = [algorithms[i]+' - '+str(round(average_ranks[i], 3)) for i in range(len(average_ranks))]

# This method computes the critical difference for Nemenyi test with alpha=0.1.
# For some reason, this method only accepts alpha='0.05' or alpha='0.1'.
cd = compute_CD(average_ranks,
                n=len(results_array),
                alpha='0.05',
                test='nemenyi')
# This method generates the plot.
graph_ranks(average_ranks,
                names=names,
                cd=cd,
                width=6,
                textspace=1.5)
plt.title(f'Friedman-Nemenyi={round(friedmanchisquare(*results_array).pvalue, 4)}\nCD={round(cd, 3)}')
plt.show()

# This method computes the critical difference for Bonferroni-Dunn test with alpha=0.05.
# For some reason, this method only accepts alpha='0.05' or alpha='0.1'.
cd = compute_CD(average_ranks,
                    n=len(results_array),
                    alpha='0.05',
                    test='bonferroni-dunn')
# This method generates the plot.
graph_ranks(average_ranks,
                names=names,
                cd=cd,
                cdmethod=0,
                width=6,
                textspace=1.5)
plt.title(f'Bonferroni-Dunn\nCD={round(cd, 3)}')
plt.show()