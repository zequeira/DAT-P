import os
import re
import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)


def load_data(file_name):
    print('FILE EXIST')
    dataDF = pd.read_csv(file_name, sep=';')
    return dataDF


if __name__ == '__main__':

    data_results = 'csnoisy.results50.csv'
    if os.path.isfile(data_results):
        data = load_data(data_results)

    # Select only the following columns:
    dataF = data[['user', 'condition', 'envCond', 'file', 'rating', 'trapQx', 'trapQxSF', ]]

    # Check number of ratings per user
    pd.DataFrame(dataF)['user'].value_counts(sort=True, ascending=True)
    # Check if users failed any trapping question
    pd.DataFrame(dataF)['trapQxSF'].value_counts()
    # Remove ratings/data from user "99" as it was a test user
    dataF = dataF[dataF.user != 99]

    # Remove the ratings corresponding to the audio files with either "Ausgezeichnet" | "Ordentlich" | "Schlecht" in
    # the name as these were the trapping questions.
    dataF = dataF[~dataF.file.str.contains("Ausgezeichnet")]
    dataF = dataF[~dataF.file.str.contains("Ordentlich")]
    dataF = dataF[~dataF.file.str.contains("Schlecht")]

    pd.DataFrame(dataF)['user'].value_counts(sort=True, ascending=True)

    dataF['id'] = range(1, dataF.shape[0]+1)

    # Create a new column containing a string representing a short version of the files' name
    dataF['fileShort'] = [re.sub(r'CH_', '', x) for x in dataF['file']]
    for idx, value in dataF['fileShort'].items():
        grps = re.search(r'^(.{3})(.{2})\_(c\d{2})', value).groups()
        dataF['fileShort'].loc[idx] = grps[0]+grps[1]+grps[2]

    dataF.user = dataF.user.astype(str)
    dataF.condition = dataF.condition.astype(str)
    dataF.file = dataF.file.astype(str)
    dataF.fileShort = dataF.fileShort.astype(str)
    dataF.rating = dataF.rating.astype(int)
    dataF.trapQx = dataF.trapQx.astype(int)
    dataF.trapQxSF = dataF.trapQxSF.astype(str)
    dataF.id = dataF.id.astype(int)

    # Save the data cleaned, ready for analysis
    dataF.to_csv('csnoisy.csv', sep=';', float_format='%.3f')
    # Save the data as *.sav file for analysis on IBM SPSS
    pyreadstat.write_sav(dataF, "csnoisy.sav")

    # Read the "Ground Truth" Laboratory (Lab) data
    df, meta = pyreadstat.read_sav("Lab.sav", apply_value_formats=True, formats_as_category=True)

    df = df.sort_values('Filename')
    # Aggregate laboratory ratings per speech degradation condition
    df_aggCond = df.groupby('cond')['Perfileavg'].mean()
    df_aggCond = df_aggCond.to_frame()

    # Calculate the Pearson correlation and Root-Mean-Squared-Error between the Laboratory and the crowdsourcing ratings
    env_cond = dataF.envCond.unique()
    # For each study session, i.e., "CSLvlx" (that corresponds to ratings collected in different environment conditions), calculate
    for environment in dataF.envCond.unique():
        dataF_env = dataF[dataF.envCond == environment]
        # Aggregate on file
        dataF_agg = dataF_env.groupby(['file', 'condition'], as_index=False)['rating'].mean()
        dataF_agg = dataF_agg.to_frame()
        dataF_agg = dataF_agg.sort_values('file')

        # Scatterplot between the Laboratory and the crwdsourcing ratings per file for each of the "CSLvlx" study session
        classes = dataF_agg.condition.unique().tolist()
        conditions = df.cond.values
        conditions = [int(x) for x in conditions]
        colours = ListedColormap(['r', 'b', 'g'])
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        scatter = plt.scatter(dataF_agg.rating.values, df.Perfileavg.values, c=conditions)
        plt.title('Laboratory vs. Crowdsourcing in {}'.format(environment))
        plt.xlabel('Crowdsourcing')
        plt.ylabel('Laboratory')
        fig.legend(title='Conditions', handles=scatter.legend_elements(num=15)[0], labels=classes, loc='right', borderaxespad=0.1)
        # plt.subplots_adjust(right=0.85)
        plt.show()

        # Code to calculate the Spearman correlation coefficient with associated p-value
        # rho, p = stats.spearmanr(dataF_agg.rating.values, df.Perfileavg.values)
        rho, p = stats.pearsonr(dataF_agg.rating.values, df.Perfileavg.values)
        rmse = sqrt(mean_squared_error(df.Perfileavg.values, dataF_agg.rating.values))
        print('Pearson correlation, Lab vs {2}: {0:.3f}, p-value:{1}; RMSE={3:.3f}'.format(rho, p, environment, rmse))

    # Calculate the Pearson correlation and RMSE per speech degradation condition between the Laboratory and
    # the crowdsourcing ratings
    for environment in dataF.envCond.unique():
        dataF_env = dataF[dataF.envCond == environment]
        # Aggregate on speech degradation condition
        dataF_agg = dataF_env.groupby('condition')['rating'].mean()
        dataF_agg = dataF_agg.to_frame()
        dataF_agg = dataF_agg.sort_values('condition')

        # Code to calculate the Spearman correlation coefficient with associated p-value
        # rho, p = stats.spearmanr(dataF_agg.rating.values, df_aggCond.Perfileavg.values)
        rho, p = stats.pearsonr(dataF_agg.rating.values, df_aggCond.Perfileavg.values)
        rmse = sqrt(mean_squared_error(df_aggCond.Perfileavg.values, dataF_agg.rating.values))
        print('Pearson correlation, Lab vs {2}: {0:.3f}, p-value:{1}; RMSE={3:.3f}'.format(rho, p, environment, rmse))


    # Representation of the Mean Opinion Scores with 95% confidence intervals per speech degradation condition
    # given by the listeners in crowdsourcing in each of the study session, i.e., CSLvl0, 1, 2, and 3.
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(x="condition", y="rating", hue="envCond", ci=95, ax=ax, data=dataF)
    plt.show()

    # ANOVA to determine if the speech degradation conditions were perceived differently in the different study
    # sessions executed under various environmental conditions
    for cond in dataF.condition.unique():
        dataF_cond = dataF[dataF.condition == cond]
        anova_rm = AnovaRM(dataF_cond, 'rating', 'user', within=['envCond'], aggregate_func='mean')
        res = anova_rm.fit()
        print('Condition: ', cond)
        print(anova_rm.fit())

        MultiComp = MultiComparison(dataF_cond['rating'], dataF_cond['envCond'])
        tukey_hsd = MultiComp.tukeyhsd().summary()
        print(tukey_hsd)

        comp_bonferroni = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
        print(comp_bonferroni[0])