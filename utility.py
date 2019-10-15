import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer, MinMaxScaler
from scipy.stats.mstats import gmean
from scipy.stats import zscore


def dataClean(df):
    # Import World Bank dgp per capita data (DEFUNCT: didnt help at all... idea was to replace countries with gdp per capita
    # from world bank dataset)

    # world_bank = pd.read_csv('tcdml1920-income-ind/world_bank_gdp.csv')
    # world_bnk_dict = {}
    # for idx, row in world_bank.iterrows():
    #     world_bnk_dict.update({row['Country Name']: row['GDPPP']})
    # Accept pandas data frame df which will be cleaned

    # List Features
    # print(list(df))

    # Countries are already clean
    # for idx, row in df.iterrows():
    #     country = row['Country']
    #     gdp_pc = world_bnk_dict[country]
    #     df.at[idx,'Country'] = gdp_pc

    # Size of City contains no Nans or 0s and is polttable... presuming ok

    # Body Height [cm] appears to be good

    # Profession needs work - 50 people has a job no one else has, lot of same jobs with different idenifying strings
    df['Profession'] = df['Profession'].fillna('unknown')

    # University Degree - removing 0 and Nan values and replacing with unknown
    df['University Degree'] = df['University Degree'].fillna('unknown')
    df['University Degree'] = df['University Degree'].replace('0','unknown')

    # Not using one hot for Degree as each level has a relationship with one another(PhD > Masters > Bachelors etc.)
    df['University Degree'] = df['University Degree'].replace('No',0)
    df['University Degree'] = df['University Degree'].replace('unknown',1)
    df['University Degree'] = df['University Degree'].replace('Bachelor',2)
    df['University Degree'] = df['University Degree'].replace('Master',3)
    df['University Degree'] = df['University Degree'].replace('PhD',4)

    # Wears Glasses is clean already

    # Gender, remove Nans and 0s
    df['Gender'] = df['Gender'].fillna('unknown')
    df['Gender'] = df['Gender'].replace('0','unknown')

    # Remove 0s with unknown
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # Clean hair colomn
    df['Hair Color'] = df['Hair Color'].fillna('unknown')
    df['Hair Color'] = df['Hair Color'].replace('0','unknown')
    df['Hair Color'] = df['Hair Color'].replace('Unknown','unknown')

    # Fill empty years with gmean - always ~2000
    mean_year = gmean(df['Year of Record'].dropna())
    df['Year of Record'] = df['Year of Record'].fillna(mean_year)
    df['Year of Record'] = df['Year of Record'].replace('0',mean_year)
    df['Year of Record'] = df['Year of Record'].replace('unknown',mean_year)

    return df

def remove_outliers(df):
    # Defunct: not fit for purpose and not used atm

    std_dev = 4.5

    df_new = df[(np.abs(zscore(df['Age'])) < std_dev)]
    df_new = df_new[(np.abs(zscore(df_new['Size of City'])) < std_dev)]
    df_new = df_new[(np.abs(zscore(df_new['Body Height [cm]'])) < std_dev)]

    return df_new

def preProcess(df_train, df_test, df_pred):
    country_replace = False

    # Task -1: Remove outliers (Defunct: can be improved but current implementation not appropiate)
    #df_train_new = remove_outliers(df_train)
    df_train_new = df_train

    # We return y's from here because of legacy...
    y_train = df_train_new.iloc[:,-1]
    y_test = df_test.iloc[:,-1]
    

    # Convert countrys to average income (Defunct: not used, suggested that you minimise this if statement in your ide)
    if country_replace:
        country_list = np.array(df_train_new['Country'].unique())
        mean_inc_count = {}
        mean_inc_all = df_train_new['Income in EUR'].mean()
        std_inc_all = df_train_new['Income in EUR'].std()
        for country in country_list:
            d = df_train_new[df_train_new['Country'] == country]['Income in EUR']
            sd = d.std()**2
            mn = d.mean()
            if np.isnan(sd):
                sd = 250000**2
            mean_inc_count.update({country:(mn,sd)})
        
        df_train_new.insert(10,'Country_std',0)
        for idx, row in df_train_new.iterrows():
            country = row['Country']
            df_train_new.at[idx,'Country'] = mean_inc_count[country][0]
            df_train_new.at[idx,'Country_std'] = mean_inc_count[country][1]

        
        for idx, row in df_test.iterrows():
            country = row['Country']
            if country in mean_inc_count.values():
                df_test.at[idx,'Country'] = mean_inc_count[country][0]
                df_test.at[idx,'Country_std'] = mean_inc_count[country][1]
            else:
                df_test.at[idx,'Country'] = mean_inc_all
                df_test.at[idx,'Country_std'] = std_inc_all
        
        for idx, row in df_pred.iterrows():
            country = row['Country']
            if country in mean_inc_count.values():
                df_pred.at[idx,'Country'] = mean_inc_count[country][0]
                df_pred.at[idx,'Country_std'] = mean_inc_count[country][1]
            else:
                df_pred.at[idx,'Country'] = mean_inc_all
                df_pred.at[idx,'Country_std'] = std_inc_all
    
    df_train_new = df_train_new.drop('Income in EUR', axis=1)
    df_test_new = df_test.drop('Income in EUR', axis=1)
    # Task 0: Group some of professions together. (Promising approach but could not get to work as wished >ToDo)
    # count = df_train_new['Profession'].value_counts()
    # df_train_new['Profession'] = np.where(df_train_new['Profession'].isin(count.index[count>4]),df_train_new['Profession'],'Other')

    # Task 1: add extra column for 2nd order polynomial for education and age (testing...)
    df_train_new['edu_sq'] = df_train_new['University Degree']**2
    df_test_new['edu_sq'] = df_test_new['University Degree']**2
    df_pred['edu_sq'] = df_pred['University Degree']**2

    df_train_new['age_sq'] = df_train_new['Age']**2
    df_test_new['age_sq'] = df_test_new['Age']**2
    df_pred['age_sq'] = df_pred['Age']**2

    # TASK 2: Convert classifications to one hot (label first)

    # Convert datafraes to numpy arrays
    df_train_new = df_train_new.to_numpy()
    df_test_new = df_test_new.to_numpy()
    df_pred_new = df_pred.to_numpy()
    
    # Label Encode categorical data - must fit to concatinated dataset as labels must be same for all dfs
    labelencoder = LabelEncoder()
    labelencoder = labelencoder.fit(np.concatenate([df_train_new[:, 1], df_test_new[:,1], df_pred_new[:,1]]))
    df_train_new[:, 1] = labelencoder.transform(df_train_new[:, 1])
    df_test_new[:, 1] = labelencoder.transform(df_test_new[:, 1])
    df_pred_new[:, 1] = labelencoder.transform(df_pred_new[:, 1])
    
    #county
    if not country_replace:
        labelencoder = labelencoder.fit(np.concatenate([df_train_new[:, 3], df_test_new[:,3], df_pred_new[:,3]]))
        df_train_new[:, 3] = labelencoder.transform(df_train_new[:, 3])
        df_test_new[:, 3] = labelencoder.transform(df_test_new[:, 3])
        df_pred_new[:, 3] = labelencoder.transform(df_pred_new[:, 3])

    # proffession
    labelencoder = labelencoder.fit(np.concatenate([df_train_new[:, 5], df_test_new[:,5], df_pred_new[:,5]]))
    df_train_new[:, 5] = labelencoder.transform(df_train_new[:, 5])
    df_test_new[:, 5] = labelencoder.transform(df_test_new[:, 5])
    df_pred_new[:, 5] = labelencoder.transform(df_pred_new[:, 5])

    labelencoder = labelencoder.fit(np.concatenate([df_train_new[:, 8], df_test_new[:,8], df_pred_new[:,8]]))
    df_train_new[:, 8] = labelencoder.transform(df_train_new[:, 8])
    df_test_new[:, 8] = labelencoder.transform(df_test_new[:, 8])
    df_pred_new[:, 8] = labelencoder.transform(df_pred_new[:, 8])

    # One hot encode labelencoded data
    if country_replace:
        onehotencoder = OneHotEncoder(categorical_features = [1, 5, 8], handle_unknown="ignore", n_values='auto')
    else:
        onehotencoder = OneHotEncoder(categorical_features = [1, 3, 5, 8], handle_unknown="ignore", n_values='auto')
    onehotencoder = onehotencoder.fit(df_train_new)
    df_train_new = onehotencoder.transform(df_train_new)
    df_test_new = onehotencoder.transform(df_test_new)
    df_pred_new = onehotencoder.transform(df_pred_new)

    df_train_new = df_train_new.toarray()
    df_test_new = df_test_new.toarray()
    df_pred_new = df_pred_new.toarray()

    # TASK 3: Scale Data appropiately
    scaler = MinMaxScaler()
    scaler.fit(df_train_new)
    df_train_new = scaler.transform(df_train_new)
    df_test_new = scaler.transform(df_test_new)
    df_pred_new = scaler.transform(df_pred_new)

    # Return all processed data
    return [y_train, y_test, df_train_new, df_test_new, df_pred_new]

def printPersonDetails(person):
    # Useful utility function to print out a person's details
    print('Instance: ', person['Instance'])
    print('Year of Record: ', person['Year of Record'])
    print('Gender: ', person['Gender'])
    print('Age: ', person['Age'])
    print('Country: ', person['Country'])
    print('Size of City: ', person['Size of City'])
    print('Profession: ', person['Profession'])
    print('University Degree: ', person['University Degree'])
    print('Wears Glasses: ', person['Wears Glasses'])
    print('Hair Color: ', person['Hair Color'])
    print('Body Height [cm]: ', person['Body Height [cm]'])
    print('Income in EUR: ', person['Income in EUR'])
    print('')

def _main_():
    # IGNORE: Just here for testing an inspection of data etc.
    train_data = pd.read_csv(
        "tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv")
    test_data = pd.read_csv(
        "tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv")

    input_data = train_data.drop('Income in EUR', axis=1).append(test_data.drop('Income', axis=1))
    world_bank = pd.read_csv('tcdml1920-income-ind/world_bank_gdp.csv')
    
    wrd_bnk_country_list = world_bank['Country Name'].unique()
    list_to_fix = ["test", 'rbrwb']

    # for country in input_data['Country']:
    #     if country not in wrd_bnk_country_list:
    #         if country not in list_to_fix:
    #             print(f"Did not find {country} in world bank list")
    #             list_to_fix.append(country)

    count = input_data['Profession'].value_counts()
    print(count)
    input_data['Profession'] = np.where(input_data['Profession'].isin(count.index[count>7]),input_data['Profession'],'Other')
    count = input_data['Profession'].value_counts()
    print(count)
    sns.distplot(input_data[input_data['Profession'] == 'Other']['Income in EUR'])

if __name__ == '__main__':
    _main_()
