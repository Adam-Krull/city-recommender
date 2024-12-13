import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from requests import get

from env import API_KEY

def acquire_data():
    '''Acquires and does basic cleaning on data. Reads locally if possible.'''
    filename = 'raw_data.csv'
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        states = get_states()
        clean_df = clean_data()
        clean_df = pd.merge(clean_df, states, how='inner', on='state')
        feature_df = collect_data(clean_df)
        feature_df.to_csv('raw_data.csv', index=0)
        final_df = pd.concat((clean_df, feature_df), axis=1)
        final_df.to_csv(filename, index=0)
        return final_df
        
def get_states(key=API_KEY):
    '''Since I get timed out by the government API, I'll select a few states.'''
    desired_states = ['Texas', 'Colorado', 'Utah', 'Idaho', 'Wyoming']
    url = f'https://api.census.gov/data/2023/acs/acs5?get=NAME&for=state:*&key={API_KEY}'
    response = get(url)
    states = pd.DataFrame(response.json())
    states.columns = states.iloc[0]
    states = states.shift(-1)
    states.columns = [col.lower() for col in states.columns]
    return states[states.name.isin(desired_states)]

def clean_data(key=API_KEY):
    '''Cleans up column and city names. Returns dataframe.'''
    url = f'https://api.census.gov/data/2023/acs/acs5/profile?get=NAME&for=place:*&key={key}'
    response = get(url)
    cities = pd.DataFrame(response.json())
    cities.columns = cities.iloc[0]
    cities = cities.shift(-1)
    cities.columns = [col.lower() for col in cities.columns]
    regexp = r'\s[a-zA-Z]*,'
    cities.name = cities.name.str.replace(regexp, ',', regex=True)
    return cities

def collect_data(cities, key=API_KEY):
    '''Collects pieces of information about all cities in the dataset.'''
    feature_list = ['DP03_0002PE', 'DP03_0027PE', 'DP03_0028PE',
                    'DP03_0029PE', 'DP03_0030PE', 'DP03_0031PE', 'DP03_0062E',
                    'DP03_0095PE', 'DP04_0046PE', 'DP04_0089E', 'DP04_0134E']
    feature_names = ['Population', 'Percent employed',
                     'Occupation (MBSA)', 'Occupation (S)',
                     'Occupation (SO)', 'Occupation (RCM)',
                     'Occupation (PT)', 'Median household income',
                     'Percent insured', 'Homeownership rate',
                     'Median home price', 'Median rent']
    values = []
    feature_string = ','.join(feature_list)
    i = 0
    for row in cities.itertuples(index=False):
        i += 1
        if (i % 100 == 0) and i > 99:
            print(f'{i} cities collected.')
        metrics = []
        pop_url = f'https://api.census.gov/data/2023/acs/acs5?get=B01003_001E&for=place:{row.place}&in=state:{row.state}&key={key}'
        pop_response = get(pop_url)
        metrics.append(pop_response.json()[1][0])
        url = f'https://api.census.gov/data/2023/acs/acs5/profile?get={feature_string}&for=place:{row.place}&in=state:{row.state}&key={key}'
        response = get(url)
        metrics.extend(response.json()[1][:-2])
        values.append(metrics)
    return pd.DataFrame(values, columns=feature_names)

def plot_all(df):
    '''Plots all features using subplots on one figure.'''
    plt.figure(figsize=(16,9))
    plt.subplot(341)
    grouped = df.groupby('State').State.count()
    plt.bar(grouped.index, grouped)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.title('# of Cities')

    plt.subplot(342)
    plt.hist(df['Population'][df['Population'] < 100_000], bins=25)
    plt.title('Population')

    plt.subplot(343)
    plt.hist(df['Percent employed'])
    plt.title('Percent employed')

    plt.subplot(344)
    plt.hist(df['Occupation (MBSA)'])
    plt.title('% in MBSA')

    plt.subplot(345)
    plt.hist(df['Occupation (S)'])
    plt.title('% in Service')
    plt.ylabel('Count')

    plt.subplot(346)
    plt.hist(df['Occupation (SO)'])
    plt.title('% in Sales or Office')

    plt.subplot(347)
    plt.hist(df['Occupation (RCM)'])
    plt.title('% in RTM')

    plt.subplot(348)
    plt.hist(df['Occupation (PT)'])
    plt.title('% in PT')

    plt.subplot(349)
    plt.hist(df['Median household income'])
    plt.title('Median household income')
    plt.ylabel('Count')

    plt.subplot(3,4,10)
    plt.hist(df['Homeownership rate'])
    plt.title('Homeownership rate')

    plt.subplot(3,4,11)
    plt.hist(df['Median home price'])
    plt.title('Median home price')

    plt.subplot(3,4,12)
    plt.hist(df['Median rent'])
    plt.title('Median rent')

    sns.despine()
    plt.tight_layout()    

def scale(df):
    '''Scales the numerical columns of the dataframe and returns only those.''' 
    feature_names = ['Population', 'Percent employed',
                     'Occupation (MBSA)', 'Occupation (S)',
                     'Occupation (SO)', 'Occupation (RCM)',
                     'Occupation (PT)', 'Median household income',
                     'Homeownership rate', 'Median home price', 'Median rent']
    scaler = MinMaxScaler()
    scaled = df.copy()
    scaled[feature_names] = scaler.fit_transform(scaled[feature_names])
    return scaled

def elbow_plot(scaled):
    '''Tries various values for n_clusters and creates an elbow plot of the result.'''
    feature_names = ['Population', 'Percent employed',
                     'Occupation (MBSA)', 'Occupation (S)',
                     'Occupation (SO)', 'Occupation (RCM)',
                     'Occupation (PT)', 'Median household income',
                     'Homeownership rate', 'Median home price', 'Median rent']
    results = []
    for i in range(2, 21):
        model = KMeans(n_clusters=i, random_state=42)
        inertia = model.fit(scaled[feature_names]).inertia_
        d = {'n_clusters': i, 'inertia': inertia}
        results.append(d)
    rdf = pd.DataFrame(results)
    plt.plot(rdf.n_clusters, rdf.inertia)
    plt.title('Elbow plot')
    plt.ylabel('Inertia')
    plt.xlabel('# of Clusters')
    sns.despine()
    plt.show()    

def create_labels(scaled, df):
    '''Creates labels from clusters and adds them to the original df.'''
    feature_names = ['Population', 'Percent employed',
                     'Occupation (MBSA)', 'Occupation (S)',
                     'Occupation (SO)', 'Occupation (RCM)',
                     'Occupation (PT)', 'Median household income',
                     'Homeownership rate', 'Median home price', 'Median rent']  
    kmeans = KMeans(n_clusters=10, random_state=42)
    klabels = kmeans.fit(scaled[feature_names]).labels_
    hier =  AgglomerativeClustering(n_clusters=10, linkage='complete')
    hlabels = hier.fit(scaled[feature_names]).labels_

    df['KMeans'] = klabels
    df['Hierarchical'] = hlabels
    return df