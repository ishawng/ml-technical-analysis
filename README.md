## Introduction

Technical analysis is a common method used to predict stock prices that has been studied since the late 1800s. Most of the recent work has focused on testing different machine learning techniques using common technical indicators [1] and using the technical indicators of correlated stocks with a hybrid Genetic Algorithm and Support Vector Machine approach [2]. This project focuses on the stocks in the S&P 500 and will use daily time-frame data from Yahoo Finance from which the technical indicators can be calculated.

## Problem Definition

In technical analysis, indicators derived from market data are used to find mathematical patterns to determine when to buy or sell a stock, with the best traders achieving a success rate greater than 70% above the market. Furthermore, this method is widely used so market forces can be modeled partially as a function of these indicators. Our project seeks to determine how machine learning can be leveraged on a combination of indicators to succeed in the market.

## Methods:

Conduct unsupervised learning on a number of stocks from the S&P 500 to separate them into clusters based on their price data over a number of years (2010-2020). To cluster based on similarity of sequences of data points, K-Means will be applied with Dynamic Time Warping (DTW) metric [4]. To evaluate its effectiveness, we will compare it with K-Means with Euclidean distance metric. 

Supervised Learning will then be conducted on each of the clusters. The papers we reviewed ([1],[2],[3]) mostly suggested SVM and Naive Bayes since they were the most successful, and we will be comparing them both. Although we have the types of indicators as the features of the dataset, we require a form of labeling to identify when a stock should be bought and sold. [1] suggested the ‘Moving Average’ Indicator for this, but this indicator is already a feature. Thus, we propose a novel approach to labeling the stock data via the peak valley algorithm [5], which maximizes the profit over the given timeframe, and will compare with [1]’s results. 

## Potential results and discussion:
 
Our group hopes to find clusters of stocks related by technical indicators, as a finding like this would increase the performance of technical analysts by allowing them to focus on certain indicators based on the stock. Furthermore, we hope that our novel profit peak valley labeling will outperform results using [1]’s labeling, since the supervised learning methods train for maximizing profit. This would reveal the importance of choosing the right type of labeling for the dataset.
 
## Timeline

We will select two models by 2/17 and 2/28. For both models, Alistair will work on data cleaning, Ishawn on data visualization, Faisal on feature reduction, and Andrew and Virinchi on model implementation. Everyone will work on the analysis and report. These steps will take until 4/5. Then, everyone will work on final analysis and video creation and submission before 4/26.

## References:

[1] Dash, R., Dash, P. K. (2016). A hybrid stock trading framework integrating technical analysis with machine learning techniques. Journal of Finance and Data Science, 2(1), 42-57. https://www.sciencedirect.com/science/article/pii/S2405918815300179.

[2] Choudhry, R., Garg, K. (2008). A hybrid machine learning system for stock market forecasting. World Academy of Science, Engineering and Technology. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.307.6153&rep=rep1&type=pdf. 

[3] Sreekumar A., Kalkur P., Moiz M. (2019). Practical Market Indicators for Algorithmic Stock Market Trading: Machine Learning Techniques and Grid Strategy. In: Shetty N., Patnaik L., Nagaraj H., Hamsavath P., Nalini N. (eds) Emerging Research in Computing, Information, Communication and Applications. Advances in Intelligent Systems and Computing, vol 906. Springer, Singapore. https://doi.org/10.1007/978-981-13-6001-5_10

[4] Amidon A. (2020, July 16). How to Apply K-means Clustering to Time Series Data. Towards Data Science, Medium. https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3

[5] GeeksforGeeks. (2021, December 22). Stock Buy Sell to Maximize Profit. https://www.geeksforgeeks.org/stock-buy-sell/
