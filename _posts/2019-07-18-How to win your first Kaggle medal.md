---
layout:     post
title:      How to win your first Kaggle competition medal
subtitle:   A guide to Kaggle machine learning competition beginners
date:       2019-07-18
author:     YC
header-img: img/post-bg-cook.jpg
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
Kaggle is a famous platform in data science. If you would like to get a job as a data scientist, a fancy Kaggle profile will strenthen you resume greatly. Here is a quora question: [How can we use Kaggle?](https://www.quora.com/How-can-we-use-kaggle) to explain more what Kaggle can do for you.
The medals you win in competitions, therefore, can be a strong evidence that you are a talented data scientist and more competent that other job applicants.

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
(3) Count missing values, decide whether to exclude, or fill with median or mean value.
(4) Use groupby to create new features with your domain knowledge.
(5) Visualize numeric features with histogram or boxplot.

**Visualization matters!!!** Most of the time, people find the 'magic' hidden in the datasets with visualization. If you saw something wierd in the figure, dig into it and find out why. Visualize the data distribution by different ways in combination with the target values can be a good approach.

After EDA, you can try training and prediction with several algorithms to have a quick look of the results based on metrics used for scoring in the competition. You can start with LightGBM, XGBosst, Deep neural network and linear regression models as they are the most popular ones in 2019 (as far as I see in Kaggle competitions). But do not spend too much time on models and their hyperparameter tuning, there are more important things for you.


### 2. Feature engineering

Feature engineering (FE) is the **MOST IMPORTANT** part in the competition, which improves your model the most as a consensus from all Kagglers. **Hyperparameter tuning** and **Ensemble** are less important in most cases. Therefore, this section will try to cover as much as I know to help Kaggle beginners to get your first medal.

**Checklist**
(1) PCA


## Tips when you get stuck
(1) Follow and participate actively the discussion in the competition forum. You can get inspirations from other peoples' EDA and even part of the solution (I do not encourage people let the magic out, as it is unfair to hardworking players).

(2) Check previous winning solutions from completed competitions. Try to find a similar one and take a close look at the champion solutions. Here are some helpful links that people have collected for the past winning solutions.

(3)







> 关键词：优化试听体验


### 参考

- [WWDC 2018 Keynote](https://developer.apple.com/videos/play/wwdc2018/101/)
- [Apple WWDC 2018: what's new? All the announcements from the keynote](https://www.techradar.com/news/apple-wwdc-2018-keynote)
- [iOS 加入「防沉迷」，macOS 有了暗色主题，今年的 WWDC 重点都在系统上](http://www.ifanr.com/1043270)
- [苹果 WWDC 2018：最全总结看这里，不错过任何重点](https://sspai.com/post/44816)
