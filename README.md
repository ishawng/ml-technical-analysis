## Predicting Stock Prices With Technical Analysis

### Proposal Video:
<video src="https://user-images.githubusercontent.com/50810904/155036324-39cf45cf-6e4d-47f0-ab16-b10ad9d3f17e.mp4" controls="controls" style="max-width: 100%;"></video>

## Introduction

Technical analysis is a common method used to predict stock prices that has been studied since the late 1800s. Most of the recent work has focused on testing different machine learning techniques using common technical indicators [1] and using the technical indicators of correlated stocks with a hybrid Genetic Algorithm and Support Vector Machine approach [2]. This project focuses on the stocks in the S&P 500 and will use daily time-frame data from Yahoo Finance from which the technical indicators can be calculated.

## Problem Definition

In technical analysis, indicators derived from market data are used to find mathematical patterns to determine when to buy or sell a stock, with the best traders achieving a success rate greater than 70% above the market. Furthermore, this method is widely used so market forces can be modeled partially as a function of these indicators. Our project seeks to determine how machine learning can be leveraged on a combination of indicators to succeed in the market.

## Data Collection

The top 50 stocks were taken from the S&P 500, and their data was pulled from Yahoo Finance API on the daily timeframe. Training data was collected from 1-1-2013 to 12-31-2020 and test data was collected from 1-1-2021 to 3-1-2022. Below is a snapshot of the type of data collected (taken from training data of AAPL). 

![Alt text](images/dataset_snapshot.png?raw=true "Figure 1: typical form of data for dataset")

The above dataset is all that is needed for unsupervised learning, however, for supervised learning, features in terms of the indicators are required.

All Indicators are some mathematical function of “date”, “open”, “high”, “low”, and “close”, “volume.” Thus the above dataset is all that is required to acquire the features.

For an example of a feature, consider one of the simplest indicators, a 34-day Simple Moving Average (SMA34). The SMA34 takes the average of the closing price of the last 34 days. Taking the SMA34 we compute our first feature as, “close” - SMA34.  If the difference is positive it implies that the closing price is higher than the SMA34 (constitutes a buy), and vice versa if negative (sell). The reason for choosing data as a float instead of a +1 for a buy and -1 for a sell, is because more information is retained with floats. To illustrate this, the plot of this difference (for training data of AAPL from 2013-2020) can be seen below.

![Alt text](images/sma34.png?raw=true "Figure 2")

However, as one can observe from the above data, as time progresses the difference becomes larger. This is expected since as AAPL’s price increases, the difference between the price and moving average increases as well. This poses a problem since this feature is not invariant to time. So any future predictions will always grow in amplitude since price increases generally, and go outside the trained data in the training dataset, leading to a worsening feature with time since training data. To fix this issue, we normalize by the 5-day Exponential Moving Average (EMA5). This indicator is selected as the exponential gives more weight to recent days within the 5 day window and better reflects current price, while getting rid of most noise that occurs from just choosing the “close” as the normalizing metric. Applying this normalization we see the new graph below. It is observed that it is largely invariant to time, and thus is a better feature.

![Alt text](images/sma34_normalized.png?raw=true "Figure 3")

In a similar way we define the rest of the 39 features for a total of 40, and are seen below. All these indicators were implemented using the python library ‘Technical Analysis Library’ [6].

 (Note: where it is mentioned ratio, it means that the value is normalized by the EMA5; MACD1 = parameters of (26,12,9); MACD2 = parameter of (60,2,2); For more information on the indicators please check investopedia) 

Features:
1. Close - EMA5 ratio
2. Close - SMA34 ratio
3. Close - SMA50 ratio
4. Close - SMA100 ratio
5. Close - SMA200 ratio
6. Close - EMA21 ratio
7. Close - EMA55 ratio
8. Close - EMA144 ratio
9. SMA34 - SMA50 ratio
10. SMA34 - SMA100 ratio
11. SMA34 - SMA200 ratio
12. SMA50 - SMA100 ratio
13. SMA50 - SMA200 ratio
14. SMA100 - SMA200 ratio
15. EMA21 - EMA55 ratio
16. EMA21 - EMA144 ratio
17. EMA55 - EMA144 ratio
18. SMA34 - EMA144 ratio
19. MACD1"s difference ratio
20. MACD1’s macd ratio
21. MACD2’s difference ratio
22. MACD2’s macd ratio
23. RSI2
24. RSI5
25. RSI14
26. RSI28
27. RSI5-RSI14
28. ATR14 ratio
29. ATR21 ratio
30. ATR50 ratio
31. ATR14 - ATR50 ratio
32. ATR21 - ATR50 ratio
33. (Volume - VOL_SMA30) / VOL_SMA30
34. Close - VWAP5 ratio
35. Close - VWAP14 ratio
36. Close - VWAP30 ratio
37. VWAP14 - SMA34 ratio
38. VWAP14 - SMA200 ratio
39. VWAP14 - EMA144 ratio
40. VWAP5 - VWAP14 ratio

Then for the labels for supervised learning, we consider 2 methods: the method from [1] which uses the MA15 (advanced by 1 day), and our novel method of peak-valley applied to Stock data Machine Learning, based on [5]. The label +1 implies a buy/hold, 0 means don’t do anything and -1 implies sell/keep cash. 

This concludes all the data collection and processing that was done to acquire the datasets,  features and labels for the unsupervised and supervised learning parts.

## Methods:

We conduct unsupervised learning on a number of stocks from the S&P 500 to separate them into clusters based on their price data over a number of years (2010-2020). More specifically, we use normalized closing prices as our features. The normalization is done by dividing each closing price by the max closing price the stock has ever attained, which will create features whose values fall in the range [0, 1]. This is a necessary step because we want to cluster stocks that follow similar trends, not necessarily stocks that have similar prices. For example, if stock A follows the same trend of highs and lows as stock B, we want them to be in the same cluster even if stock A’s share price is consistently $100 greater than stock B’s (vertical shifts in trends should not affect clustering results).

To cluster based on similarity of sequences of data points, K-Means will be applied with Dynamic Time Warping (DTW) metric [4]. To evaluate its effectiveness, we will compare it with K-Means with Euclidean distance metric and a Gaussian Mixture Model.

[UPDATE] Supervised Learning will then be conducted on each of the clusters. The papers we reviewed ([1],[2],[3]) mostly suggested SVM and Naive Bayes since they were the most successful, and we will be comparing them both. Although we have the types of indicators as the features of the dataset, we require a form of labeling to identify when a stock should be bought and sold. [1] suggested the ‘Moving Average’ Indicator for this, but this indicator is already a feature. Thus, we propose a novel approach to labeling the stock data via the peak valley algorithm [5], which maximizes the profit over the given timeframe, and will compare with [1]’s results.
 

## Unsupervised Learning Results & Discussion:

Our unsupervised portion of the project consisted of clustering the top 50 S&P 500 stocks based on normalized daily closing prices. The goal of doing this was to group stocks with similar trends in hopes that certain sets of technical indicators best predict the performance of each cluster.

The first clustering method we used was K-Means with Euclidean distance. Running the elbow method revealed 4 clusters as being a good choice for the dataset:

INSERT IMAGES

Most of the clusters ended up being fairly mixed, with one cluster of outliers. This is likely because Euclidean distance isn’t a very good metric when dealing with time-based data, as it is invariant to time shifts. That is, if two time series are similar, but shifted by some number of time steps, Euclidean distance fails during clustering.

To solve this problem, we next used K-Means with Dynamic Time Warping (DTW). As motivated by [4], DTW is a better metric to use when clustering time series because it can measure the similarity in trend between two time series even if the similarity does not begin at the same time. The resulting elbow method plot for this strategy revealed 6 clusters as being a good choice for the dataset:

INSERT IMAGES

The clusters that resulted from DTW are slightly better than the ones that resulted from Euclidean distance. The two stocks that were outliers in Euclidean distance are now integrated into clusters, instead of being separate in their own cluster. Also, from a visual analysis, the stocks that are clustered together tend to have similar properties. For example, Tesla, Facebook, NVIDIA, Adobe, and Netflix are all clustered together, and are all technology-oriented companies.

Lastly, we used a Gaussian Mixture Model (GMM) for clustering to see if the soft and non-spherical clustering capabilities of GMM would improve our results. Running the GMM with 4 clusters gave the following:

INSERT IMAGES

The GMM performed similarly to K-Means with Euclidean distance. GMM suffers from the same shortcomings as K-Means with Euclidean distance. Again, we can see that there are two stocks that are outliers in their own cluster, while the other three clusters have a large mixture of stocks.

Based on the results of the three clustering algorithms that we tried, we have decided to use K-Means with Dynamic Time Warping as our primary method of clustering the stocks.

## References:

[1] Dash, R., Dash, P. K. (2016). A hybrid stock trading framework integrating technical analysis with machine learning techniques. Journal of Finance and Data Science, 2(1), 42-57. [https://www.sciencedirect.com/science/article/pii/S2405918815300179](https://www.sciencedirect.com/science/article/pii/S2405918815300179)

[2] Choudhry, R., Garg, K. (2008). A hybrid machine learning system for stock market forecasting. World Academy of Science, Engineering and Technology. [https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.307.6153&rep=rep1&type=pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.307.6153&rep=rep1&type=pdf)

[3] Sreekumar A., Kalkur P., Moiz M. (2019). Practical Market Indicators for Algorithmic Stock Market Trading: Machine Learning Techniques and Grid Strategy. In: Shetty N., Patnaik L., Nagaraj H., Hamsavath P., Nalini N. (eds) Emerging Research in Computing, Information, Communication and Applications. Advances in Intelligent Systems and Computing, vol 906. Springer, Singapore. [https://doi.org/10.1007/978-981-13-6001-5_10](https://doi.org/10.1007/978-981-13-6001-5_10)

[4] Amidon A. (2020, July 16). How to Apply K-means Clustering to Time Series Data. Towards Data Science, Medium. [https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3](https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3)

[5] GeeksforGeeks. (2021, December 22). Stock Buy Sell to Maximize Profit. [https://www.geeksforgeeks.org/stock-buy-sell/](https://www.geeksforgeeks.org/stock-buy-sell/)

[6] https://technical-analysis-library-in-python.readthedocs.io/en/latest/
