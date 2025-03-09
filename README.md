# Overview

This project collects information about almost 2,000 cities in 5 states of interest (Texas, Idaho, Wyoming, Colorado, Utah). The goal is to use this information to find new places to live/visit based on places I enjoy! The project achieves this by clustering the cities together into 10 unique clusters.

# Data

The data is acquired from the US Census Bureau via their [API](https://api.census.gov/data.html). All information is pulled from the most recent ACS 5-Year survey tables (2023). I specifically pulled information from the data profiles and subject tables. I collected 11 metrics about all cities in the 5 states. The data dictionary is seen below.

| Metric | Code | Value |
| ------ | ---- | ----- |
| Total population | B01003_001E | Estimate |
| Employment | DP03_0002PE | Percentage |
| Occupation (business, science, arts) | DP03_0027PE | Percentage |
| Occupation (service industry) | DP03_0028PE | Percentage |
| Occupation (sales and office) | DP03_0029PE | Percentage |
| Occupation (natural resources, construction, maintenance) | DP03_0030PE | Percentage |
| Occupation (production, transportation) | DP03_0031PE | Percentage |
| Median household income | DP03_0062E | Estimate |
| Homeownership | DP04_0046PE | Percentage |
| Median home price | DP04_0089E | Estimate |
| Median rent | DP04_0134E | Estimate |

# Data cleaning

The data was clean when it arrived. I dropped cities from the dataset that had incomplete records. Of the original 3,000 cities I found, I was able to keep records for almost 2,000 of them. Column names were cleaned up and redundant columns were dropped.

# EDA

Histograms were created for each feature. Main takeaways from the histograms were:
- Texas has the majority of cities in the dataset, with over 1,100! This makes sense, because the population of Texas is much greater than the others.
- The vast majority of cities have small populations: 5,000 people or fewer. The big cities were the outliers!
- Employment seems to be pretty equally distributed between the five types listed in the data dictionary
- Median household income appears concentrated between 50-80,000 dollars per year
- Median home price is almost exclusively between 0-500,000 dollars
- Median rent centers around 1,000 dollars per month, with a tail that extends into higher prices

# Preprocessing

The data was transformed before the modeling stage using min-max scaling. This was done to put all features on a level playing field. The goal is to prevent the clustering models from becoming biased in favor of certain features.

# Modeling

A series of K-means models were used to create an elbow plot, in an effort to determine the optimal number of clusters. The selected value based on the chart was 10 clusters.

K-means and hierarchical clustering models were fit to the dataset and used to predict the labels. Both sets of labels were explored. It was determined hierarchical clustering resulted in intuitive and interpretable clusters. The hierarchical clustering labels were kept as the "achievement" of the project.
