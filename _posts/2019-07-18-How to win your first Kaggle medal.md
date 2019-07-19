---
layout:     post
title:      How to win your first Kaggle competition medal
subtitle:   A guide to Kaggle machine learning competition beginners
date:       2019-07-18
author:     YC
header-img: img/atm_image.png
catalog: true
tags:
    - Machine learning
    - Kaggle
    - Feature engineering
---

## Prologue
### What is Kaggle
Kaggle is an online community of data scientists and machine learners owned by Google. Machine learning competitionsare most famous feature of Kaggle. Data scientists and learners compete to build the best algorithm for problems posted by companies to win medals or money. Suppose you already have some coding experience and mathmatical background but are still new to Kaggle machine learning competitions.

### What does a medal mean to you
Kaggle is a famous platform in data science. If you would like to get a job as a data scientist, a fancy Kaggle profile will strenthen you resume greatly. Here is a quora question: [How can we use Kaggle?](https://www.quora.com/How-can-we-use-kaggle) to explain more what Kaggle can do for you. The medals you win in competitions, therefore, can be a strong evidence that you are a talented data scientist and more competent than other job applicants. A competition on Kaggle can be a good opportunity to you by completing a real-world data project to [enter data science jobs](https://www.quora.com/Will-doing-well-on-Kaggle-get-me-an-entry-level-data-science-job), and you can even [get a job right after competition!](https://www.kaggle.com/general/45686)

This blog will guide you through the whole process of a Kaggle competition and help you to win a medal, including the strategies that almost everyone uses to win the prize and a checklist for you when you are out of ideas during the competition.


### Which competition to choose

There are mainly three kinds of competitions, the **featured**, **research** and **knowledge**. The first two are relatively competitive as they have  money prizes for top players and the **knowledge** is for skill practice. If you want to take it easy for your first competition and do not have much spare time, then I suggest you to go for the **knowlege competition** such as [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) or [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

But if you want a fierce battle to prove yourself on your CV, then you should definitely go for the **featured** ones. Compared to **research** competitions, **featured** ones have more participants and thus more competitive. But do not be afraid, the chance to win a medal is still high as the medal system is depending on the size of the competition.  

![](img/medal-system.jpg)

Then, there will be mainly 3 kinds of data format in competitions that you need to choose based on your career interest and skill set. The numeric data, image data, and text data. Choose the one fits you best!

>关键词：官方防沉迷最为致命

## How to complete a Kaggle competition

Here comes the main course. This section will explain the whole procedure to go through a competition, mainly focus on numeric data format as it is the mainstream with widest applications.

### 1. Exploratory data analysis (EDA)

EDA should always be the first thing to do once you download the datasets. Sometimes you can find nice kernels contributed by other participants in which they have performed EDA for the community. But it is always nicer that you can do it yourself to obtain some unique observations from the datasets.

**Checklist**
(1) Identify numeric and categorical features.
(2) Count unique values of categorical features.
(3) Count missing values, decide whether to exclude, or fill with 0, median or mean value.
(4) Use groupby to create new features with your domain knowledge.
(5) Visualize numeric features with histogram or boxplot.

**Visualization matters!!!** Most of the time, people find the 'magic' hidden in the datasets with visualization. If you saw something wierd in the figure, dig into it and find out why. Visualize the data distribution by different ways in combination with the target values can be a good approach.

After EDA, you can try training and prediction with several algorithms to have a quick look of the results based on metrics used for scoring in the competition. You can start with LightGBM, XGBosst, Deep neural network and linear regression models as they are the most popular ones in 2019 (as far as I see in Kaggle competitions). But do not spend too much time on models and their hyperparameter tuning, there are more important things for you.


### 2. Feature engineering

Feature engineering (FE) is the **MOST IMPORTANT** part in the competition, which improves your model the most as a consensus from all Kagglers. **Hyperparameter tuning** and **Ensemble** are less important in most cases. Therefore, this section will try to cover as much as I know to help Kaggle beginners to get your first medal. The techniques listed below are not always useful but you have to try before you can tell whether it will work or not. Check the list if there is anything you have not tested in the competition.

**Checklist**
- (1) Principal Component Analysis (PCA):
A dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, which trade a little accuracy for simplicity. This technique is extremely useful when dealing with complex datasets, which allows faster training and removes less useful features. But it will be difficult to interpret the new features generated by PCA. Read [this PCA post](https://towardsdatascience.com/a-step-by-step-explanation-of-principal-component-analysis-b836fb9c97e2) for more details.
- (2) Target-encoding:
A common method to convert categorical features into numerical. Read [this post](https://towardsdatascience.com/why-you-should-try-mean-encoding-17057262cd0) for more details. **Be aware of target leaking!!!** Leave the encoding outside when perform crossvalidation.
- (3) One-hot encoding:
Covert categorical features into a sparse matrix that can be provided to ML algorithms. Let's say you have a categorical feature with 4 kinds like "1", "2", "3" and "4", but this will confuse the ML algorithm that Kind '3' and kind '2' are more similar than kind '4' and kind '1' (as 3-2 < 4-1), which is obviously wrong. You can use pandas.get_dummies to process the categorical features with more than 2 unique values. [Read this post](https://medium.com/@michaeldelsole/what-is-one-hot-encoding-and-how-to-do-it-f0ae272f1179) for more details.
- (4) Aggregations and statistics:
Calculate mean, mode,frequency, min or max values of one or several features to create informative new features with your domain knowledge. This may sounds easy but will take a lot of time from you as the combinations are infinite. Most helpful new features come to you by this approach. Do not just throw missing values away, counting the number of missing values can also be a new feature.
- (5) Remove noise:
Visualize data to detect potential **outliers** and try to remove them to see effects (Outliers may be determined by percentile or other logics depending on your wild guess)Also you can try to **round decimal numbers** to less decimals and see effects. Use **[ceil]**(https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ceil.html#numpy.ceil) or **[floor]**(https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.floor.html) functions to set upper or lower limit to numbers can be another way to remove noise.
- (6) Transformation:
You can perform log transform, square, cubic, root, absolute, sin, tanh or all the mathematical transformation to your raw data. But remember, monotonic transformation has little effect on decision tree-based algorithms (Someone says it changes the way of binning a bit but I think it is not worthy to try)
- (7) Binarization:
Binning is to make the model more robust and prevent overfitting at the cost of performance. In my view, this works poorly to numeric features and never be helpful. However, for categorical features with some values of low frequencies, merge some low frequency categories can be helpful.
- (8) Feature interaction
You can try to multiply, plus, minus or devide between different features, which will make sense to create new features by catching their potential relationships. It can be difficult to craft new features if they are anonymous and too many combinations are waiting for you if you have no idea what they mean. I have seen discussions many times that Neural network can catch such feature interactions for you once it have enough data, [see this post](https://stats.stackexchange.com/questions/323954/do-neural-networks-capture-relationships-between-features), that is why it is absolutely worthy to include neural network as one of your models in the competition.
- (9) Time data
ML algorithms usually cannot interpret time data correctly and you should do a bit transformation. There are too many things you can do with time data. For instance, you can extract new features like time difference, moving average values, Business hours or not, Weekend or not, season of the year and so on based on real world logic. I encourage you to read more post of feature engineering for time data depending on your specific problems. Here is [an example post](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/) for further reading.
- (10) Feature scaling
Do not forget to try scaler to make variance of the features in the same range. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) are the most widely used scalers and have improved my model a lot in my case.
- (11) More feature engineering methods:
What I list here are the methods I have not verified myself but seen recommedations from many other people. It may worthy to try if you have already tried solutions above but still struggle to make a breakthrough. [Automated Feature Engineering](https://towardsdatascience.com/automated-feature-engineering-for-predictive-modeling-d8c9fa4e478b), saves your time and can get unexpected progress. [TF-IDF](https://galaxydatatech.com/2018/11/19/feature-extraction-with-tf-idf/), which stands for Term Frequency – Inverse Document Frequency. People have been used it for text data feature extraction, but some recent competitions show this can also be applied to categorical features.

### 3. Hyperparameter tuning

After you are totally out of feature engineering ideas and think its time to optimize your models, you can spend your time on hyperparameter tuning. Otherwise, new features added will require you to perform the tuning again. (You may see the performance of your model decreases even if a powerful new feature is feeded to your algorithm without tuning again). In order to save your time on hyperparameter tuning in a time-crunched competition, I would suggest you doing the following things.

**Checklist**
- (1) As most of the players are using similar algorithms, go to the competition forum to get the hyperparameters from a good kernel and tune based on it.
- (2) In order to know if a new feature works or not. Everytime a new feature is added, **set a random number seed** for the reproducibility and a **3-fold cross validation** to accelerate the training and prediction. More than often, you have to try hundreds of new feature engineering during the competition. It is very important you check your new features in an efficient way with several combinations of hyperparameters tuning.
- (3) Use [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to search for the optimal hyperparameters before you go sleep. It is a time-consuming step that you will never want it to occupy your productive daytime.
- (4) Alternatively, if you already know the effects of the hyperparameters to control overfitting or underfitting, you can tune them manually. This can save you more time and it is also a good chance for you to understand the mechanisms underneath the algorithms. (PS, I even tried [Bayersian optimization](https://github.com/fmfn/BayesianOptimization) but eventually found tuning hyperparameters manually save my time the most)



### 4. Model ensemble

Model ensembling is a very powerful and popular technique that almost all the Kaggle winners use it more or less. But before you going to far into those voting, stacking or blending things (I spent too much time into ensembling but only get little outcome compared to feature engineering), I would like to remind you one fundamental thing. **Ensemble only work best when you have many diverse models**. However, it would be really time-consuming to train and optimize 30 different models to get a Frankenstein ensemble, like [this 1st solution in a competition](https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335). As far as I know, such a complicated ensemble is unlikely to be implemented in industry and will cost a lot of time to build unless you are experienced enough.

**Checklist**
- (1) Try as many diverse models as you can, such as linear regression, gradient boosting trees (LightGBM and XGBoost), neural network and K-nearest neighbors. You can also create different inputs to those models and ensemble them as different models.
- (2) Each of the model should have a decent performance. It is hard to explain what is a decent performance, but for example, adding a model with accuracy less than 50 % will be highly likely to decrease the ensemble accuracy. You can try to use only some of your best models and check the performance.
- (3) As far as I observed, a weighted blending can be sufficient to win a competition. Even [the guy won the champion](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89003#latest-521279) in a 8800 teams competition used a simple ensemble by weighted blending of neural network and LightGBM. So **feature engineering** should still be your priority!
- (4) Do the ensemble at the end of competition.

An example of how ensemble improve the accuracy.


## Tips when you get stuck
- (1) Follow and participate actively the discussions in the competition forum. You can get inspirations from other peoples' EDA and even part of the solution (I do not encourage people let the magic out, as it is unfair to hardworking players).

- (2) Check previous winning solutions from completed competitions. Try to find a similar one and take a close look at the champion solutions. Some top players even shared their completed solutions on their Github repositories after competitions. Here are some helpful links that people have collected about the past winning solutions.[A colletion from Farid Rashidi](https://faridrashidi.github.io/kaggle-solutions/); [A post contributed by Kaggler SRK](https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions)

- (3) Read more articles about how people win their Kaggle competitions and other techniques used. Here I list some links that I found quite helpful. [A feature engineering book written by Alice Zheng](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/); [An ensemble guide post](https://mlwave.com/kaggle-ensembling-guide/);


## Summary

A Kaggle competition is not easy for everyone but can be the best opportunity for you to learn how to become a professional data scientist. Where there is a will there is a way. I hope you can win your first Kaggle medal and start your career in Data Science after reading this blog. Hard work pays off!!!




### Author information
I am a PhD from Aarhus University in Denmark. I participated my first Kaggle machine learning comepetition with 8800 teams(Santander Customer Transaction Prediction) and ranked 100th in the end without a data science background. If you find any mistake in the article or think something important is not mentioned, you are welcome to contact me by caiyunpeng1989@gmail.com to help more Kaggle beginners.

### References

- [Fundamental Techniques of Feature Engineering for Machine Learning](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)
- [用python参加Kaggle的些许经验总结](https://www.jianshu.com/p/32def2294ae6)
- [iOS 加入「防沉迷」，macOS 有了暗色主题，今年的 WWDC 重点都在系统上](http://www.ifanr.com/1043270)
- [苹果 WWDC 2018：最全总结看这里，不错过任何重点](https://sspai.com/post/44816)
