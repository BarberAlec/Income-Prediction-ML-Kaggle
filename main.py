import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

from utility import dataClean, preProcess, printPersonDetails
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm
import scipy.stats as stats 
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def main(arg1=None,arg2=None,dest_file=None):
    # Load dataset
    train_data = pd.read_csv(
        "tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv")
    pred_data = pd.read_csv(
        "tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv")

    # Print head and colomn headers to get sense of data
    # print(train_data.head())
    # print(train_data.columns.values.tolist())

    # Split data into Train and Test set
    X_train_set, X_test_set, _, _ = train_test_split(train_data, train_data.iloc[:,-1],test_size=0.2)
    
    # Firstly remove all values with negative incomes... (removed...)
    #train_data = train_data.drop(train_data[train_data['Income in EUR'] < 1].index)

    # Fit Gamma function to Training Income (Approach I took where I matched out income distribution with in dist.)
    #train_alpha, _, train_beta = stats.invgamma.fit(train_data['Income in EUR'],floc=0)

    # Seperate Features from output
    X_train_set = X_train_set.drop('Instance',axis=1)
    X_test_set = X_test_set.drop('Instance',axis=1)

    # Drop Empty Income and useless Instance colomns
    X_pred = pred_data.drop('Income', axis=1)
    X_pred = X_pred.drop('Instance', axis=1)
    
    # Clean data (function definition found in utility.py) and process (scale and one hot encode)
    X_train_set_clean = dataClean(X_train_set)
    X_test_set_clean = dataClean(X_test_set)
    X_pred_clean = dataClean(X_pred)

    # Must preProcess all the data together becauseof legacy code :p
    y_train, y_test, X_arr, X_test_set_arr, X_pred_arr = preProcess(X_train_set_clean, X_test_set_clean, X_pred_clean)
    


    # Fit model (comment out applicable lines as required)

    # The old reliable
    # reg = LinearRegression().fit(X_arr,y_train)

    # COuld never get to converge and give good ish answer
    # reg = Lasso(tol=0.02).fit(X_arr,y_train)

    # Ehh, not great
    # reg = MLPRegressor().fit(X_arr,y_train)

    # Very slow and pretty aweful results
    # reg = svm.SVR(cache_size=7000).fit(X_arr,y_train)

    # SVM/SVR (same as above but a little quicker)
    # reg = svm.LinearSVR().fit(X_arr,y_train)

    # least squares with l2 regularization (quick and quite good)
    # reg = Ridge(alpha=0.1).fit(X_arr,y_train)
    #reg = RidgeCV(alphas=np.arange(0.05,1,0.02)).fit(X_arr,y_train)

    # Random Tree forest (very effective but very slow and memory intensive)
    # if arg1 and arg2:
    #     reg = RandomForestRegressor(max_depth=int(arg1),n_estimators=int(arg2)).fit(X_arr,y_train)
    # else:
    #     reg = RandomForestRegressor(n_estimators=200).fit(X_arr,y_train)#max_depth=20
    
    # Extremely effective and quite quick - Best results, also found iterations above ~20'000 are not useful
    reg = CatBoostRegressor(iterations=20000,task_type="GPU").fit(X_arr,y_train)
    

    # Make predictions and print scores
    y_pred = reg.predict(X_pred_arr)
    scores = [reg.score(X_arr,y_train), reg.score(X_test_set_arr,y_test)]
    print(f"Train accuracy: {scores[0]}")
    print(f"Test accuracy: {scores[1]}")
    print(f"Train accuracy: {sqrt(mean_squared_error(y_train,reg.predict(X_arr)))}")
    print(f"Test accuracy: {sqrt(mean_squared_error(y_test,reg.predict(X_test_set_arr)))}")
    

    # # Make new csv with output data
    out_data = pd.read_csv(
        "tcdml1920-income-ind/tcd ml 2019-20 income prediction submission file.csv")
    out_data_arr = out_data.values

    # Refit results to fit correct gamma function (one of my crazy approaches where i fit invgammadist
    # to the income dist and then forced the outward income to fit the same dist.... limitied success)
    #y_pred = transform_data_2_invgamma(y_pred,train_alpha,train_beta)

    out_data_arr[:, 1] = y_pred[:]

    # Initially thought that negitive values for income were bad, then found out that the are accpetable.... 
    #out_data_arr = remove_negitive_income(out_data_arr)

    # Save file
    if dest_file:
        np.savetxt(dest_file, out_data_arr, "%i", delimiter=",")
    else:
        np.savetxt("pred_file.csv", out_data_arr, "%i", delimiter=",")
    
    # returns scores for a script that runs different models on different threads
    # turned out to be not that useful as 16 Gb of ram apparently is only enough to train 4 models at once
    return scores

def transform_data_2_invgamma(x,alp,beta):
    # Transfomr data to fit given distribution.... worth a shot
    x_size = len(x)

    # Define local parametres
    time_step = 15
    dist_x = np.arange(0,4500000,time_step)
    pdf = stats.invgamma.pdf(dist_x,alp,scale=beta)
    
    # Create volume increments (think integrations over delta intervals)
    pdf_delta = pdf*time_step

    # Create a freq list for each income value (difference defines currently unallocated realisations)
    realisations_per_delta = np.round(pdf_delta*x_size)
    difference = int(x_size - sum(realisations_per_delta))

    # Create vector of output results in increasing number
    out_arr = np.array([])
    for i in range(len(realisations_per_delta)):

        # While still difference left, add 1 to each freq position...
        if realisations_per_delta[i] == 0 and difference != 0:
            realisations_per_delta[i] = 1
            difference -= 1
        
        # Add income values to output array
        val = time_step/2 + time_step*i
        out_arr = np.concatenate((out_arr,np.array([val]*int(realisations_per_delta[i]))),axis=0)

    # Map results to output vector
    index_order_x = np.argsort(x)
    result = out_arr[index_order_x]

    sns.distplot(result)
    plt.show()

    return result


def remove_negitive_income(arr):
    arr[arr<0] = 1000
    return arr


def epected_val(dist_1,dist_2,x_dist,time_step):
    # Designed for my stats approach, given a distribution, calculate the expected value
    dist_1_clean = np.nan_to_num(dist_1)
    dist_2_clean = np.nan_to_num(dist_2)

    if (np.isnan(dist_1).any() or np.isnan(dist_2).any()):
        print("Input has nans")

    result = np.multiply(dist_1_clean, dist_2_clean)
    result = result/(sum(result)*time_step)
    if (np.isnan(result).any()):
        print(dist_1)
        print(dist_2)
        plt.subplot(2,1,1)
        sns.lineplot(x_dist,dist_1)
        plt.subplot(2,1,2)
        sns.lineplot(x_dist,dist_2)
        plt.show()

    mean = sum(np.multiply(result,x_dist*time_step))
    return mean


def stats_approach():
    # Crazy apprach where I build a pointwise distribution for income for each country and profession seperately.
    # Then I would multiply dists to get output distribution where I would find the expected value
    # Only implemented proffessions and Contry, limitied success
    train_data = pd.read_csv(
        "tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv")
    pred_data = pd.read_csv(
        "tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv")

    train_data = train_data.drop(train_data[train_data['Income in EUR'] < 2].index)
    train_data = train_data.drop(train_data[train_data['Size of City'] > 6000000].index)

    # Seperate Features from output
    #X = train_data.drop('Income in EUR', axis=1)
    X = train_data.drop('Hair Color', axis=1)
    X = X.drop('Wears Glasses',axis=1)
    X = X.drop('Instance',axis=1)

    y = train_data.iloc[:, -1]

    #X_pred = pred_data.drop('Income', axis=1)
    #X_pred = X_pred.drop('Profession', axis=1)
    X_pred = pred_data.drop('Hair Color', axis=1)
    X_pred = X_pred.drop('Wears Glasses',axis=1)
    X_pred = X_pred.drop('Instance', axis=1)
    
    # Clean data (function definition found in utility.py) and process (scale and one hot encode)
    X_clean = dataClean(X)
    X_pred_clean = dataClean(X_pred)
    X_comb = X_clean.append(X_pred_clean)

    unique_countries = X_comb['Country'].unique()
    unique_professions = X_comb['Profession'].unique()

    # Create a dict for idx
    country_dict = {}
    prof_dict = {}
    for i, country in enumerate(unique_countries):
        country_dict.update({country: i})
    for i, pro in enumerate(unique_professions):
        prof_dict.update({pro: i})

    step_size = 20
    dist_x = np.arange(0,350000,step_size)

    country_point_pdf = np.zeros([len(unique_countries), int(350000/step_size)])
    profession_point_pdf = np.zeros([len(unique_professions), int(350000/step_size)])

    # Complete country pdfs
    for idx,country in enumerate(unique_countries):
        if len(X_clean[X_clean['Country'] == country]) == 0:
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(X_clean[X_clean['Country'] == 'Austria']['Income in EUR'],floc=0)
            pdf_country = stats.gamma.pdf(dist_x,fit_alpha,scale=fit_beta)
            country_point_pdf[idx,:] = pdf_country
            continue

        if len(X_clean[X_clean['Country'] == country]) < 2:
            pdf_country = stats.norm.pdf(dist_x,loc=X_clean[X_clean['Country'] == country]['Income in EUR'],scale=100000)
            country_point_pdf[idx,:] = pdf_country
            continue
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(X_clean[X_clean['Country'] == country]['Income in EUR'],floc=0)
        pdf_country = stats.gamma.pdf(dist_x,fit_alpha,scale=fit_beta)
        country_point_pdf[idx,:] = pdf_country

    # Complete profession pdfs
    for idx,prof in enumerate(unique_professions):
        if len(X_clean[X_clean['Profession'] == prof]) == 0:
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(X_clean[X_clean['Profession'] == 'unknown']['Income in EUR'],floc=0)
            pdf_prof = stats.gamma.pdf(dist_x,fit_alpha,scale=fit_beta)
            profession_point_pdf[idx,:] = pdf_prof
            continue

        if len(X_clean[X_clean['Profession'] == prof]) < 2:
            pdf_prof = stats.norm.pdf(dist_x,loc=X_clean[X_clean['Profession'] == prof]['Income in EUR'],scale=100000)
            profession_point_pdf[idx,:] = pdf_prof
            continue
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(X_clean[X_clean['Profession'] == prof]['Income in EUR'],floc=0)
        pdf_prof = stats.gamma.pdf(dist_x,fit_alpha,scale=fit_beta)
        profession_point_pdf[idx,:] = pdf_prof


    y_pred = np.zeros([len(X_pred_clean.index), 1])
    # Calulate predicitons for test data
    for idx, row in X_pred_clean.iterrows():
        country = row['Country']
        prof = row['Profession']

        c_dist = country_point_pdf[country_dict[country],:]
        p_dist = profession_point_pdf[prof_dict[prof],:]
        y_pred[idx] = epected_val(c_dist,p_dist,dist_x,step_size)



    out_data = pd.read_csv(
        "tcdml1920-income-ind/tcd ml 2019-20 income prediction submission file.csv")
    out_data_arr = out_data.values

    print(out_data_arr[:, 1].shape)
    print(y_pred.shape)
    out_data_arr[:, 1] = np.squeeze(np.nan_to_num(y_pred))
    np.savetxt("pred_file.csv", out_data_arr, "%i", delimiter=",")

    # plt.subplot(4,1,1)
    # print(unique_countries[10])
    # sns.lineplot(dist_x,country_point_pdf[10,:])
    # plt.subplot(4,1,2)
    # print(unique_countries[11])
    # sns.lineplot(dist_x,country_point_pdf[11,:])
    # plt.subplot(4,1,3)
    # print(unique_countries[12])
    # sns.lineplot(dist_x,country_point_pdf[12,:])
    # plt.subplot(4,1,4)
    # print(unique_countries[13])
    # sns.lineplot(dist_x,country_point_pdf[13,:])
    # plt.show()

if __name__ == '__main__':
    # If laucnhed externally: then you can define hyper parametres for models
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main()
    #stats_approach()

    # Warning: some random testing stuff below;can ignore...


    # x_plot = np.arange(0,250000,20)

    # train_data = pd.read_csv(
    #     "tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv")
    # data = dataClean(train_data)
    
    
    # #plt.show()

    # fit_alpha, fit_loc, fit_beta=stats.gamma.fit(data[data['Country'] == 'Norway']['Income in EUR'],floc=0)
    # p = stats.gamma.rvs(fit_alpha, scale=fit_beta, size=10000)
    # plt.subplot(3, 1, 1)
    # #sns.distplot(data[data['Country'] == 'Norway']['Income in EUR'])
    # pdf_country = stats.gamma.pdf(x_plot,fit_alpha,scale=fit_beta)
    # sns.lineplot(x_plot,pdf_country)
    

    # fit_alpha_prof, fit_loc_prof, fit_beta_prof = stats.gamma.fit(data[data['Profession'] == 'pest control worker']['Income in EUR'],floc=0)
    # #p = stats.gamma.rvs(fit_alpha, scale=fit_beta, size=10000)
    # plt.subplot(3, 1, 2)
    # #sns.distplot(data[data['Profession'] == 'pest control worker']['Income in EUR'])
    # pdf_prof = stats.gamma.pdf(x_plot,fit_alpha_prof ,scale=fit_beta_prof)
    # sns.lineplot(x_plot,pdf_prof)


    # plt.subplot(3, 1, 3)
    # result = np.multiply(pdf_prof,pdf_country)
    # result = result/(sum(result)*20)


    # print(np.argmax(result)*20)
    # print(sum(np.multiply(result,x_plot*20)))
    # sns.lineplot(x_plot,result)
    # plt.show()

    # print(data['Body Height [cm]'][0:3])
    # print(data['Income in EUR'].median())
    # print(data['Income in EUR'].mean())
