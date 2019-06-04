# Pitchfork Reviews Hypothesis Testing and Statistical Analysis


Introduction
---------------

Our project is aimed towards the editorial team at Pitchfork. We examine reviews
from January 1999 to January 2017 to determine whether ratings are statistically
different within different categories - more specifically, we will examine
differences in review scores between music genres, between record labels, between
music labelled "Best New Music" and between reviewers. Our goal is to provide
insight on possible biases within their reviews.


Installation
---------------

In order to get the Jupyter notebooks to work, you will have to download the
Pitchfork Reviews Sqlite Database first [here on Kaggle](https://www.kaggle.com/nolanbconaway/pitchfork-data)

Make sure you place the unzipped file under the project root folder and name it
`database.sqlite`. A description of the DB can be found in the `pitchfork-db.png`
file.

![Done](https://i.giphy.com/media/9Jcw5pUQlgQLe5NonJ/giphy-downsized.gif)


How-To
---------

Once you have setup the Pitchfork database, you are good to go through each of
the questions we tried to answer during our project. Those questions are detailed
in a distinct Jupyter Notebook inside the project :

- Question1 - Bias by Genre
- Question2_Label
- Question3 - Score Threshold
- Question4_ScorePrediction

An extra python file named 'DocumentClassification.py' contains our Naive Bayes
Document Classification python class used in Question 4.

Have a good read and hope you will enjoy our analysis
