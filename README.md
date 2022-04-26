## Predicting Stock Prices With Technical Analysis

### Final Video:
<video src="https://user-images.githubusercontent.com/50810904/165395714-4f71c5d4-e686-4005-b4c4-6906a2513862.mp4" controls="controls" style="max-width: 100%;"></video>

<!--
### Proposal Video:
<video src="https://user-images.githubusercontent.com/50810904/155036324-39cf45cf-6e4d-47f0-ab16-b10ad9d3f17e.mp4" controls="controls" style="max-width: 100%;"></video>
-->

## Introduction

Technical analysis is a common method used to predict stock prices that has been studied since the late 1800s. Most of the recent work has focused on testing different machine learning techniques using common technical indicators [1] and using the technical indicators of correlated stocks with a hybrid Genetic Algorithm and Support Vector Machine approach [2]. This project focuses on the stocks in the S&P 500 and will use daily time-frame data from Yahoo Finance from which the technical indicators can be calculated.

## Problem Definition

In technical analysis, indicators derived from market data are used to find mathematical patterns to determine when to buy or sell a stock, with the best traders achieving a success rate greater than 70% above the market. Furthermore, this method is widely used so market forces can be modeled partially as a function of these indicators. Our project seeks to determine how machine learning can be leveraged on a combination of indicators to succeed in the market.

## Data Collection

The top 50 stocks were taken from the S&P 500, and their data was pulled from Yahoo Finance API on the daily timeframe. Training data was collected from 1-1-2013 to 12-31-2020 and test data was collected from 1-1-2021 to 3-1-2022. Below is a snapshot of the type of data collected (taken from training data of AAPL). 

![Alt text](images/dataset_snapshot.png?raw=true "Figure 1: typical form of data for dataset")

The above dataset is all that is needed for unsupervised learning, however, for supervised learning, features in terms of the indicators are required.

All Indicators are some mathematical function of “date”, “open”, “high”, “low”, “close” and “volume.” Thus the above dataset is all that is required to acquire the features.

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

To cluster based on similarity of sequences of data points, K-Means will be applied with Dynamic Time Warping (DTW) metric [4]. To evaluate its effectiveness, we will compare it with K-Means with Euclidean distance metric and a Gaussian Mixture Model. We use K-Means with DTW instead of Euclidean distance because Euclidean distance is not a very good metric when dealing with time-series data, as it is invariant to time shifts. That is, if two time series are similar, but shifted by some number of time steps, Euclidean distance fails during clustering. As motivated by [4], DTW is a better metric to use when clustering time series because it can measure the similarity in trend between two time series even if the similarity does not begin at the same time.

Supervised learning was conducted on the top 50 stocks to make purchase decisions using the most common technical indicators as features. The first supervised learning method that was used was Gaussian Naive Bayes (GNB). Before GNB could be used on the data, the data was cleaned up. Many of the features are calculated based on a range of data, so the values for those features were NaN if enough data was not available. To illustrate this, consider the SMA34 feature. The SMA34 feature is the average of the closing price of the past 34 days. For the first 33 days considered in the training set, the SMA34 can not be calculated, so its value is NaN. GNB does not work with NaN values for features, so all NaN values in the training and testing features were replaced with zero before using with GNB. 

To get a baseline, GNB was first applied without performing feature reduction with principal component analysis (PCA). Next, the optimal number of components to keep with PCA was determined by starting with one component and incrementing by one to 40 components and running GNB on all stocks for each number of components. The key metrics used to evaluate the different GNB cases was the accuracy and final balance for each labeling scheme, peak-valley and MA15. After running GNB on all the stocks, the stocks were clustered using the clusters from the unsupervised learning. The same process was performed on the clustered stocks to determine whether a different number of PCA components is ideal and whether there is any increase in the final balance of the stocks. Next, SVM will be used, and the results will be compared with GNB. 

 

## Unsupervised Learning Results & Discussion:

Our unsupervised portion of the project consisted of clustering the top 50 S&P 500 stocks based on normalized daily closing prices. The goal of doing this was to group stocks with similar trends in hopes that certain sets of technical indicators best predict the performance of each cluster.

The first clustering method we used was K-Means with Euclidean distance. Running the elbow method revealed 4 clusters as being a good choice for the dataset:

![Alt text](images/knn_elbow.PNG?raw=true "Figure 4")

![Alt text](images/knn.PNG?raw=true "Figure 5")

Most of the clusters ended up being fairly mixed, with one cluster of outliers. Again, this is likely because Euclidean distance isn’t a very good metric when dealing with time-series data.

To solve this problem, we next used K-Means with Dynamic Time Warping (DTW). The resulting elbow method plot for this strategy revealed 6 clusters as being a good choice for the dataset:

![Alt text](images/dtw_elbow.PNG?raw=true "Figure 6")

![Alt text](images/dtw.PNG?raw=true "Figure 7")

The clusters that resulted from DTW are slightly better than the ones that resulted from Euclidean distance. The two stocks that were outliers in Euclidean distance are now integrated into clusters, instead of being separate in their own cluster. Also, from a visual analysis, the stocks that are clustered together tend to have similar properties. For example, Tesla, Facebook, NVIDIA, Adobe, and Netflix are all clustered together, and are all technology-oriented companies.

Lastly, we used a Gaussian Mixture Model (GMM) for clustering to see if the soft and non-spherical clustering capabilities of GMM would improve our results. Running the GMM with 4 clusters gave the following:

![Alt text](images/gmm.PNG?raw=true "Figure 8")

The GMM performed similarly to K-Means with Euclidean distance. GMM suffers from the same shortcomings as K-Means with Euclidean distance. Again, we can see that there are two stocks that are outliers in their own cluster, while the other three clusters have a large mixture of stocks.

Based on the results of the three clustering algorithms that we tried, we have decided to use K-Means with Dynamic Time Warping as our primary method of clustering the stocks.

## GNB - Supervised Learning Results & Discussion:

### GNB - All Stocks

First, GNB was used with no feature reduction to establish a baseline that is then compared with the results of GNB with feature reduction. For this, GNB was run on the top 50 stocks from the S&P 500, and the predicted labels for each stock were used to simulate how an account that invested in the stock would fare during the testing period. The buy-hold account is the simplest type of investing where a stock is bought and held throughout a period and is used to determine how much the peak-valley and MA15 labeling methods improve. For the figures below and all subsequent figures, “unit account” implies that the account starts at 1, and the final amount is a ratio of the starting amount. For example, in the below figure for buy-hold, AMZN (in brown), has a final amount of ~2. This means that the account grew 2x the original value. So if we had $100, the final value would be $200. Thus, we see that “unit account” is invariant of the initial amount invested. 

First, each stock was considered individually, so a separate GNB classifier was fitted to the training data of each stock. The test labels for each stock were determined by using the respective GNB classifier for each stock, and the accuracy and final balance of all the stocks were averaged which are shown below.

![Alt text](images/GNB_all.png?raw=true "Figure 9")

|                       | Buy-hold | Peak-valley |    MA15    |
|:---------------------:|:--------:|:-----------:|:----------:|
|    Average Accuracy   |    N/A   |  0.5002069  | 0.70351724 |
| Average Final Balance |1.19225756|  1.10091563 | 1.08027871 |

As can be observed from the above table for accuracy and final balance, the MA15 has significantly higher accuracy than peak-valley, but it still yields less profit on average than both buy-hold and peak-valley. This is because the peak-valley labeling method is much more aggressive than the MA15. With peak-valley, if one were to get 100% accuracy, it would imply the algorithm got the correct prediction for every single day. However, with MA15, it needs to be correct roughly every 15 days (as per the MA15), and thus accuracy tends to be higher with MA15. However, the drawback with MA15 is that even with 100% accuracy its potential profit is still minor compared to peak-valley. Thus 50% accuracy with peak-valley results in more profit than 75% accuracy with MA15. Thus, labeling plays a critical role in the performance for stock trading machine learning. However, both labeling schemes are worse than the buy-hold strategy, so there is no advantage to using either peak-valley or MA15 as opposed to the buy-hold strategy.

To try and improve the final balance of the labeling methods, PCA was considered. To determine the optimal number of components to use with PCA, a sweep across all number of possible components, 1 to 40 components, was done with all 50 stocks with the results shown below. A similar methodology as above was used where each PCA was applied to each stocks training data individually, and a GNB classifier was fitted to the modified training data. 

![Alt text](images/PCA_sweep_all.png?raw=true "Figure 10")

Based on the results, the average final balance is maximized with the peak-valley labeling method when PCA is used with 4 components. It is interesting to note that MA15 doesn’t vary significantly with PCA, at least compared with peak-valley. Based on these results, GNB was run again using PCA with 4 components, and the results are shown below. 

|                       | Buy-hold |Peak-valley|   MA15   |
|:---------------------:|:--------:|:---------:|:--------:|
|    Average Accuracy   |    N/A   | 0.51551724|0.7382069 |
| Average Final Balance |1.19225756| 1.156911  |1.06094153|

From the table above, it can be seen peak-valley has become much more competitive with buy-hold, with it having a new final balance of ~1.16 vs. the without PCA version of ~1.10, which is about a 50% increase in performance compared to the without PCA version [50% from comparing only profit => (0.15 - 0.1) / 0.1]. As a result, peak-valley now outperforms buy-hold in a good amount of stocks. Thus, GNB with peak-valley labeling and PCA with 4 components was found to be the most beneficial strategy. 

After it was determined that 4 components maximizes the average final balance, the components created with PCA were examined to determine which of the 40 features contributed the most to each feature. Any feature with an absolute value component score greater than 0.1 was considered as contributing to a component. This analysis was performed on the data for each stock, and for all stocks, the RSI features contributed the most to all components, and therefore had the most influence. It is surprising to see that RSI played such a big role in all components. However, this might be expected since most traders use the RSI as a metric for predicting price movement. Thus, besides just performance improvement, PCA offers interesting insight into most vital indicators for predicting the stock market. 

### GNB - All Stocks with Combined Training Data

Even though PCA increased the final balance of peak-valley, it is still less than the final balance achieved using buy-hold by a decent amount, so some other strategy must be considered to try and match the performance of the buy-hold strategy. One of the main limitations with out dataset is the number of samples available for training. To include Facebook in the stocks being considered, stock data was collected from 1-1-2013 which is Facebook's first full year in the S&P500. Due to this constraint, additional training data cannot be collected, so other approaches need to be considered to artifcally increase the size of the training set. The approach that was considered is combining the training data of all 50 stocks into a single training dataset, and then fitting a single GNB classifier to the combined training dataset. The same GNB classifier is then used to make predictions for each stock's testing dataset. First, a GNB classifier was fitted to the combined training dataset without applying PCA. The average accuracy and final balance results using the single GNB classifier are shown below.

|                       | Buy-hold |Peak-valley|    MA15    |
|:---------------------:|:--------:|:---------:|:----------:|
|    Average Accuracy   |    N/A   | 0.4937931 | 0.70717241 |
| Average Final Balance |1.19225756| 1.0951549 | 1.09125909 |

Even with combining the training data, the final balance of peak-valley and MA15 are not close to the final balance achieved with the buy-hold strategy. In addition, there is no significant difference between combining all the training data and using a single GNB classifier as opposed to fitting an individual GNB classifier for each stock. The MA15 final balance only increased by 0.01, and the peak-valley final balance was actually lower with a 0.006 reduction. However, the final balance did improve before when PCA was performed, so optimal number of components was determined by sweeping across across all number of possible components. For this, PCA was applied to the combined training dataset of all 50 stocks, and a single GNB classifier was fitted to the modified training data. The accuracy and final balance seen below is the average of all 50 stocks with predictions made using the same GNB classifier.

![Alt text](images/PCA_sweep_all_comb.png?raw=true "Figure 11")

Based on the results shown in the above figure, both the peak-valley and MA15 labeling schemes achieved their highest average final balance at 17 components with the peak-valley final balance reaching close to 1.2 which is the final balance achieved by buy-hold. Since the peak-valley final balance was close to the buy-hold final balance, PCA with 17 components was applied to the combined training dataset, and the average final balance and final balance of each stock are shown below.

![Alt text](images/GNB_all_comb_pca.png?raw=true "Figure 12")

![Alt text](images/PCA_17_comb_final_bal.png?raw=true "Figure 13")

|                       | Buy-hold |Peak-valley|    MA15   |
|:---------------------:|:--------:|:---------:|:---------:|
|    Average Accuracy   |    N/A   | 0.51289655| 0.6317931 |
| Average Final Balance |1.19225756| 1.1995076 | 1.16381517|

Using PCA with 17 components significantly increased the average final balance of both peak-valley and MA15 compared to both without PCA and when seprate GNB classifiers were used for each stock. Even though the MA15 accuracy dropped, the final balance increased significantly, as the final balance finally went above 1.1 to 1.16. In addition, the peak-valley final balance was finally higher than buy-hold's which shows that the labeling scheme can outperform the buy-hold strategy. Looking at the individual stocks, the MA15 labeling scheme resulted in the highest final balance for some stocks, but for the most part, either buy-hold or peak-valley resulted in the highest final balance which reflects the results seen in the table. The negative aspect of both peak-valley and buy-hold strategies is that they are riskier than MA15. The lowest final balance with MA15 is around 0.8 for Intel (INTC), but both buy-hold and peak-valley result in final balances around 0.5 for Paypal (PYPL) which makes these strategies extremely risky for some stocks. This shows that even though MA15 has a lower average final balance than both buy-hold and peak-valley, there are some stocks for which using MA15 is more beneficial.

### GNB - Clustered Stocks

While the peak-valley labeling scheme finally achieved a higher average final balance than the buy-hold strategy, the improvement is only approximately 0.007 which translates to an increase of $7 if the original investment was $1000. To try and improve the performance even more, incorporating the clustering results from the unsupervised section was considered. In the unsupervised section, the stocks were clustered using the DTW metric with the theory that the clustered stocks have common elements which can be leveraged to increase the performance in the supervised learning. To test this, the stocks were split into clusters determined using k-means with DTW metric as shown below.

- Cluster 1: MA, AMZN, NEE, DHR, GOOG, UNH, MSFT, ACN, V, HD, COST, NKE, CRM, AVGO, AAPL, LOW, LLY, TMO, GOOGL
- Cluster 2: CVX, KO, WFC
- Cluster 3: BAC, ABT, ABBV, MCD, DIS, INTC, CSCO, JPM, CMCSA, QCOM, BRK-B, LIN
- Cluster 4: XOM, T, VZ
- Cluster 5: JNJ, PG, PFE, PEP, WMT, MRK
- Cluster 6: TSLA, FB, NVDA, ADBE, INTU, NFLX

For each cluster, the training datasets for all stocks in the cluster were combined, and a GNB classifier was fitted for each cluster using the cluster's combined training dataset. The same approach as before was used with fitting a GNB classifier without PCA and then determining the optimal number of PCA components for each cluster. The average final balance for each cluster without performing PCA is shown below.

|                           | Buy-hold |Peak-valley|    MA15  |
|:-------------------------:|:--------:|:---------:|:--------:|
|Cluster 1 Avg Final Balance|1.23951547|1.12055809 |1.12632436|
|Cluster 2 Avg Final Balance|1.55885621|1.29262168 |1.06721071|
|Cluster 3 Avg Final Balance|1.16322689|1.18224253 |1.00554879|
|Cluster 4 Avg Final Balance|1.20210187|1.15626126 |1.01956352|
|Cluster 5 Avg Final Balance|1.08442302|0.98865121 |1.05772243|
|Cluster 6 Avg Final Balance|1.13855429|1.08826255 |1.13502758|

From the table above, there are some interesting results to point out. For cluster 3, the peak-valley labeling scheme already outperforms buy-hold without any PCA, and the average final balance should increase after PCA is performed based on previous observations. This shows that for cluster 3, peak-valley should be the preferred method for making decisions. Another observation is that MA15 is currently better than peak-valley for clusters 1, 5 and 6, and it remains to be seen which labeling method will perform better after PCA is applied. The optimal number of PCA components for each cluster was determined by using the same approach as before by performing a sweep across all possible components and choosing the number of components that maximized average final balance. The final balance charts for each cluster, and optimal number of compoennts are shown below.

![Alt text](images/PCA_sweep_1.png?raw=true "Figure 14")

![Alt text](images/PCA_sweep_2.png?raw=true "Figure 15")

![Alt text](images/PCA_sweep_3.png?raw=true "Figure 16")

![Alt text](images/PCA_sweep_4.png?raw=true "Figure 17")

![Alt text](images/PCA_sweep_5.png?raw=true "Figure 18")

![Alt text](images/PCA_sweep_6.png?raw=true "Figure 19")

|                     |Number of Components|Final Balance|
|:-------------------:|:------------------:|:-----------:|
|Cluster 1 Peak-Valley|10                  |1.25021420547|
|Cluster 1 MA15       |19                  |1.15609241653|
|Cluster 2 Peak-Valley|21                  |1.45483231419|
|Cluster 2 MA15       |14                  |1.43985909669|
|Cluster 3 Peak-Valley|11                  |1.19820607275|
|Cluster 3 MA15       |20                  |1.13041225388|
|Cluster 4 Peak-Valley|2                   |1.30175344541|
|Cluster 4 MA15       |19                  |1.15809102279|
|Cluster 5 Peak-Valley|3                   |1.15793025173|
|Cluster 5 MA15       |10                  |1.15010856411|
|Cluster 6 Peak-Valley|18                  |1.28712220730|
|Cluster 6 MA15       |18                  |1.39985067218|

Based on the results from the table, the optimal number of components for each cluster are as follows (ordered from 1 to 6): 10, 21, 11, 2, 3, 18. One interesting note is that for cluster 6, both peak-valley and MA15 reached their highest average balance at 18 components, and the final balance for MA15 was actually higher than peak-valley. Cluster 6 is the only cluster where MA15 was higher than peak-valley and shows that MA15 is better for some stocks. In addition, the final balances of peak-valley and MA15 were very close for cluster 2, 3 and 5 which again shows that peak-valley is not always better than MA15. With the optimal number of components determined for each cluster, PCA was applied to the datasets for each cluster, and a GNB classifier was fitted for each cluster.

|                           | Buy-hold |Peak-valley|   MA15   |
|:-------------------------:|:--------:|:---------:|:--------:|
|Cluster 1 Avg Final Balance|1.23951547|1.25021421 |1.08142691|
|Cluster 2 Avg Final Balance|1.55885621|1.45483231 |1.32736437|
|Cluster 3 Avg Final Balance|1.16322689|1.19820607 |1.04668446|
|Cluster 4 Avg Final Balance|1.20210187|1.30175345 |1.1393762 |
|Cluster 5 Avg Final Balance|1.08442302|1.15793025 |1.04169596|
|Cluster 6 Avg Final Balance|1.13855429|1.28712221 |1.39985067|

For most of the clusters, peak-valley resulted in the higher average final balance, but there were two exceptions. For cluster 2, buy-hold had the highest final balance which was also the case before PCA was applied. For cluster 6, both peak-valley and MA15 outperformed buy-hold, but the MA15 final balance was higher than peak-valley making it the preferred labeling scheme for the stocks in cluster 6. While the results in the table are useful, it cannot be compared to the results without clustering in its current state because the performance of individual stocks is not being tracked. Therefore, the final balance of each stock using the single GNB classifier for all stocks was subtracted from the final balance of each stock trained with the GNB classifier for its cluster to help determine whether there was any improvement, and the results are shown below for each labeling scheme.

![Alt text](images/Stock_comp_peak.png?raw=true "Figure 20")

![Alt text](images/Stock_comp_MA15.png?raw=true "Figure 21")

In the figures above, a positive difference means that using clustering and training a GNB classifier by combining the training data for each cluster results in a higher final balance than using a single GNB classfier for all stocks. For peak-valley, the total improvement by using clustering was 2.09 which is approximately 0.042 per stock. On the other hand, the total improvement for MA15 was -0.92 which is approximately -0.018 per stock. Overall, clustering improved the final balance for most stocks with the peak-valley improving the most, but for some stocks MA15 is still better. For example, Facebook (FB) had a negative difference when using clustering with peak-valley but had a positive difference when using MA15. Therefore, MA15 is still useful for certain stocks, but peak-valley is better overall.


## SVM - Supervised Learning Results & Discussion:

### SVM - All Stocks with Combined Training Data

The improvement with PCA wasn't sufficient for GNB, as a result we decided to collate the training data in order to augment the amount of training data that the models had to work with. The result was both improved efficiency and performance - to the extent that it was very sensible to implement the same process in the next supervised method that we were going to test. Hence we decided against creating an identical inefficiency for the sake of comparison and instead from the start ran the SVM model on all the training data available. Like before, first a GNB classifier was fitted to the combined training dataset without applying PCA. The final balance results using the single SVM classifier are shown below.

|                       | Buy-hold |Peak-valley|    MA15    |
|:---------------------:|:--------:|:---------:|:----------:|
| Average Final Balance |1.19225756| 1.1539549 | 1.11355909 |

Same as with GNB even after combining the training data, the final balance of peak-valley and MA15 are not close to the final balance achieved with the buy-hold strategy. In addition, there is no significant difference between combining all the training data and using a single GNB classifier as opposed to fitting an individual GNB classifier for each stock. However, the final balance did improve before PCA was performed, so the optimal number of components was determined by sweeping across all numbers of possible components as was the case with GNB. For this, PCA was applied to the combined training dataset of all 50 stocks, and a single GNB classifier was fitted to the modified training data. The accuracy and final balance seen below is the average of all 50 stocks with predictions made using the same SVM classifier.

![Alt text](images/SVM1-afb-.png?raw=true "Figure 22")

Based on the results shown in the above figure, both the peak-valley and MA15 labeling schemes achieved their highest average final balance at 7 components - significantly lower than with GNB at 17 components - with the peak-valley final balance reaching close to 1.2 which is the final balance achieved by buy-hold. Since the peak-valley final balance was close to the buy-hold final balance, PCA with 7 components was applied to the combined training dataset, and the average final balance and final balance of each stock are shown in the below figures.

![Alt text](images/SVM2.png?raw=true "Figure 23")

![Alt text](images/SVM3.png?raw=true "Figure 24")

|                       | Buy-hold |Peak-valley|    MA15   |
|:---------------------:|:--------:|:---------:|:---------:|
|    Average Accuracy   |    N/A   | 0.52319655| 0.7095931 |
| Average Final Balance |1.19225756| 1.1962076 | 1.09601517|

Using PCA with 7 components significantly increased the average final balance of peak-valley but not MA15 (both increased when used with GNB). In our case both the accuracy and final balance of MA15 dropped. Unfortunately - and perhaps more importantly even with PCA and collation of the data we were not able to achieve a greater success rate with peak-valley compared to buy-hold. 

Looking at the individual stocks, the MA15 labeling scheme resulted in the highest final balance for some stocks, but for the most part, either buy-hold or peak-valley resulted in the highest final balance which reflects the results seen in the table - similar to GNB. The negative aspect of both peak-valley and buy-hold strategies is that they are riskier than MA15 - again similar to GNB, the overall characteristics of the methods and their behavior with stock movements remained similar. The lowest final balance with MA15 is around 0.8 for Intel (INTC), but both buy-hold and peak-valley result in final balances around 0.5 for Paypal (PYPL) which makes these strategies extremely risky for some stocks. This shows that even though MA15 has a lower average final balance than both buy-hold and peak-valley, there are some stocks for which using MA15 is more beneficial. Overall, peak-valley had an approximately 62% higher final balance when compared with MA15.

### SVM - Clustered Stocks

To try and improve the performance even more, incorporating the clustering results from the unsupervised section was considered. In the unsupervised section, the stocks were clustered using the DTW metric with the theory that the clustered stocks have common elements which can be leveraged to increase the performance in the supervised learning. To test this, the stocks were split into clusters determined using k-means with DTW metric as shown below.

- Cluster 1: MA, AMZN, NEE, DHR, GOOG, UNH, MSFT, ACN, V, HD, COST, NKE, CRM, AVGO, AAPL, LOW, LLY, TMO, GOOGL
- Cluster 2: CVX, KO, WFC
- Cluster 3: BAC, ABT, ABBV, MCD, DIS, INTC, CSCO, JPM, CMCSA, QCOM, BRK-B, LIN
- Cluster 4: XOM, T, VZ
- Cluster 5: JNJ, PG, PFE, PEP, WMT, MRK
- Cluster 6: TSLA, FB, NVDA, ADBE, INTU, NFLX

For each cluster, the training datasets for all stocks in the cluster were combined, and a SVM classifier was fitted for each cluster using the cluster's combined training dataset. We used a slightly more condensed method than before - in order to remain efficient, limiting our comparison to only clusters trained using PCA. The average final balance for each cluster after performing PCA is shown below.

|                           | Buy-hold |Peak-valley|    MA15  |
|:-------------------------:|:--------:|:---------:|:--------:|
|Cluster 1 Avg Final Balance|1.12391547|1.23335809 |1.13142436|
|Cluster 2 Avg Final Balance|1.55885621|1.32042168 |1.13381071|
|Cluster 3 Avg Final Balance|1.20212689|1.16314253 |1.07814879|
|Cluster 4 Avg Final Balance|1.20210187|1.16316126 |1.07816352|
|Cluster 5 Avg Final Balance|1.08442302|0.08635121 |1.10142243|
|Cluster 6 Avg Final Balance|1.13855429|1.13916255 |1.20602758|


From the table above, there are some interesting results to point out. Training the stocks on the dynamic time warp clusters, we found very similar relative results to Gausssian naive bayes. While Buy-hold and Peak-valley outperformed MA15 on most clusters, with buy-hold having the highest final balance in 75% of said cases, MA15 did outperform both buy-hold and peak-valley in clusters 5 & 6. The optimal number of PCA components was determined by using the same approach as before by performing a sweep across all possible components and choosing the number of components that maximized average final balance. The final balance charts for each cluster are shown in the below figures.


![Alt text](images/SVMc1.png?raw=true "Figure 25")

![Alt text](images/SVMc2.png?raw=true "Figure 26")

![Alt text](images/SVMc3.png?raw=true "Figure 27")

![Alt text](images/SVMc4.png?raw=true "Figure 28")

![Alt text](images/SVMc5.png?raw=true "Figure 29")

![Alt text](images/SVMc6.png?raw=true "Figure 30")

Unlike with the GNB the results are obviously fast more mixed, there is no clear winner as each of the methods was did have some success - however it can be confidently stated that neither MA15 nor peak-valley beat buy-hold overall, and in most cases peak-valley did outperform MA15. 


## GNB vs SVM - Supervised Learning Comparison Results & Discussion:

### GNB w/PCA vs SVM w/PCA

Comparing the two supervised learning methods we were able to determine that GNB was superior overall. Dividing the results further GNB saw consistently better performance when trained on clusters. For the peak-valley method the performance was better throughout all clusters and often by a significant (>0.05) margin, but either little improvement or slightly worse performance for some clusters using buy-hold and MA15. However it should be noted the sum of differences shows that once again the gains of GNB MA15 outweighs the relative gains of SVM MA15 - hence we can safely assume that the relatively conservative nature of MA15 or perhaps just improved performance of GNB would make the gains by SVM minimal in comparison - i.e. high upside with relatively less downside. Relative difference in performance is shown in the below figure - green represents where GNB beats SVM, red represents the percentage by which SVM beats GNB.

![Alt text](images/SVM5.png?raw=true "Figure 31")

In the chart in the below figure - where relative performance of GNB and SVM are compared using the single classifier trained on all collated stock data the difference was negligible for the peak-valley method but GNB once again performed markedly better for MA15 - again by a significant (>0.05) margin. It can therefore be safely concluded that GNB after performing PCA and DTW clustering has the highest performance and average final balance of any of the other methods tested including buy-hold.

![Alt text](images/SVM4.png?raw=true "Figure 32")




## References:

[1] Dash, R., Dash, P. K. (2016). A hybrid stock trading framework integrating technical analysis with machine learning techniques. Journal of Finance and Data Science, 2(1), 42-57. [https://www.sciencedirect.com/science/article/pii/S2405918815300179](https://www.sciencedirect.com/science/article/pii/S2405918815300179)

[2] Choudhry, R., Garg, K. (2008). A hybrid machine learning system for stock market forecasting. World Academy of Science, Engineering and Technology. [https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.307.6153&rep=rep1&type=pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.307.6153&rep=rep1&type=pdf)

[3] Sreekumar A., Kalkur P., Moiz M. (2019). Practical Market Indicators for Algorithmic Stock Market Trading: Machine Learning Techniques and Grid Strategy. In: Shetty N., Patnaik L., Nagaraj H., Hamsavath P., Nalini N. (eds) Emerging Research in Computing, Information, Communication and Applications. Advances in Intelligent Systems and Computing, vol 906. Springer, Singapore. [https://doi.org/10.1007/978-981-13-6001-5_10](https://doi.org/10.1007/978-981-13-6001-5_10)

[4] Amidon A. (2020, July 16). How to Apply K-means Clustering to Time Series Data. Towards Data Science, Medium. [https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3](https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3)

[5] GeeksforGeeks. (2021, December 22). Stock Buy Sell to Maximize Profit. [https://www.geeksforgeeks.org/stock-buy-sell/](https://www.geeksforgeeks.org/stock-buy-sell/)

[6] https://technical-analysis-library-in-python.readthedocs.io/en/latest/
