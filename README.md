# Motivation
Leafly is the world’s largest cannabis website. The website allows consumers to browse for strains that might match their preferences (Eg. Sativa, Indica, or Hybrid) and gather information about the effects (Eg. Euphoric, Relaxed, Sleepy, etc.) and flavors (Eg. Earthy, Citrus, Sweet) of each strain. Users of the website are able to leave ratings and reviews on strains and dispensaries. This makes it incredibly user-friendly for any new or returning patients to explore strains that are most close to their liking and locate highly-rated dispensaries in the area.

While the website is great for exploratory analysis of cannabis products, the website does not recommend strains to users based on its users’ history. There is a beta version of a ’Similar Strains’ feature on the website which seems to take only content-based features into account. The cannabis industry is rapidly growing both in terms of medicinal and recreational use. Because of this, there was (and still is) a need for improvement in the ability of websites like Leafly to match users with strains they might like. The aim in this project is to create a recommendation system that incorporates user data via collaborative filtering. Since Leafly has a large database of both products and users’ reviews, the goal was to leverage the data in order to create a better model. Another goal of the project was to construct another version of ’Similar Strains’ by clustering strains according to flavor and effects. In summary, the goal of this project was to be able to recommend specific strains to users using information gathered from both the Leafly website and a Kaggle cannabis data set.
# Data
In order to build the collaborative filtering model, the Leafly website had to be scraped for reviews and ratings left by users. Using the Chrome extension Web Scraper (https://webscraper.io/), a data set with 25,675 observations was assembled, where each observation recorded text data associated with information about the reviewer, the name of the strain being reviewed, the review left by a user (body of text), and a rating for that strain (integer from 1 to 5). Due to certain complications with the design of the website, only 1319 out of the 2351 total unique strains were able to be scraped and roughly 30% of the reviews for each strain. The scraped data also failed to include the larger category of strain (Sativa, Indica, or Hybrid). Using a complete data set found on Kaggle (which included content information about each strain), this information was recoverable and the strains were labeled with their corresponding types (Sativa, Indica, or Hybrid). The effects and flavors of each strain were also merged as variables from the kaggle data set. In total, 9 variables were left in the final data set, which are described in Table 1.

For this project, the following .csv files were used:

• cannabis.csv: This Kaggle dataset contains the features of 2350 unique strains of marijuana. The features include the type, average user rating, effects, flavor and description.

• UserRatingsReviews.csv: The scraped Leafly dataset that includes derived ratings and reviews of strains from users. Each observation contains the username, strain, the derived rating, the derived review, a user ID, a strain ID, strain type, effects, and flavors.

Table 1: Variables in the data set UserRatingsReviews.csv

|Variable|Description|
|---|---|
|username|The observation reviewer|
|strain|Name of strain|
|rating|Rating of the strain, a number out of 5|
|review|Full review text|
|userID|The observation reviewer, given as a numerical value|
|strainID|Name of strain, given as a numerical value starting at 1|
|Type|Is sativa|
|Effects|The effects of a particular strain on users|
|Flavors|The flavors of a particular strain|

The data sets were first cleaned up in order to make it easier for analysis and merging. Due to complications with the web scraping, there were random entries of ’null’ for user ratings as well as empty review texts. Consequently, any observation that had a null rating and/or review was deleted. Next, each user and strain was assigned to a unique integer ID starting from 1. Only users who had left at least 3 reviews were considered. This was done after the first attempt at building a collaborative filtering model which included all users. Since a majority of users in the data set left only one review, including these users in the first attempt of the model proved to be detrimental to the accuracy of the model (less than 3 percent) because the resulting matrix was so sparse. The threshold requirement was manually cross validated for number of reviews per user and found to have at least 3 ratings that optimized the accuracy of the final model on the validation set. Next, two copies of the data set were made. The goal was to build additional content based models (boosting, random forest, etc) with the intention of later blending these models with the collaborative filtering model. After converting the texts in these two copies to two separate corpus, they were cleaned using standard text cleaning techniques: lower casing letters, removing any punctuation, stemming the document, etc.

# Analytics
The data was split with 84% of the observations in the training set, 4% for tuning the collaborative filtering model, 4% for blending, and 8% in the testing set. Overall, 6 blending models were run: Random Forest, CART, Logistic Regression, Boosting, Linear Discriminant Analysis, and Linear Regression. Because users rated strains on a scale from 1 to 5, the models were trained to predict 1 through 5 as categorical variables. Before removing users who had fewer than 3 ratings, the average was close to 5. After removing these users, the baseline model predicted 4 for each strain because the average rating for each strain tended to be between 3.5 and 4.5, with an average of 4.3 overall. The baseline model performed with an accuracy of 35.44% on the test set. Of the content based models, Random Forest performed the best with a test-set accuracy of 49.87%. Knowing the flavors and effects of each strain gave nearly a 15% increase in predictive power.

Next, the collaborative filtering model was built. In the first initial attempt before removing users with less than 3 ratings, cross validation helped reach an optimal k value of 1. Additionally, it was observed that the corresponding accuracy of this k value was below 3%. It could be concluded that the data matrix for collaborative filtering must have been too sparse. After removing infrequent reviewers, the model was retrained and found again the optimal k value to be 1. This suggested that, according to the data, there really might only be one archetypal user. The cross validated accuracy for this k value on the validation set was much better than the previous model, but still poor compared to the baseline with an accuracy of 19%. Surprisingly, the test set accuracy using the newer models was much higher than that of the validation set (28.10%). Nevertheless, the model was blended with the content based models. The idea was that the blended model should perform at worst with an accuracy equal to that of the best content based model. The resulting accuracy of the blended model was indeed 49%.

Table 2: Accuracy of each model on test set

|Model|Accuracy|
|---|---|
|Random Forest|49.87%|
|CART|49.62%|
|Blended|49.11%|
|LDA|48.61%|
|Logistic Regression|48.35%|
|Boosting|47.85%|
|Baseline|35.44%|
|Linear Regression|35.02%|
|Collaborative Filtering|28.10%|

In light of the collaborative filtering model performing less well than imagined, the next best option for recommending strains to users would be to cluster the strains together. While the clustering itself does not take in any user data, what this approach can do is recommend related strains to a user given a strain that the user is known to like. Three different sets of hierarchical clusters were built, using strain flavors, effects, and both. Clustering by flavor only gave 49 clusters, by effects only gave 20, and by both gave 49.

For example, here is a sample strains clustered by the algorithm. It is somewhat intuitive why they were clustered together as many of them have overlapping flavor characteristics.

Table 3: Sample of strains in cluster 3 (grouped by flavor)

|Strain|Flavors|
|---|---|
|Alien-Stardawg|Diesel,Spicy,Herbal,Sage|
|Bedford-Glue|Pepper,Pine,Sage|
|Big-Band|Sweet,Spicy,Herbal,Sage|
|Biker-Leblanc|Spicy,Herbal,Sage,Earthy|
|Kushage|Earthy,Sage,Spicy,Herbal|
|Blowfish|Spicy,Herbal,Pepper,Sage|
|Bloodhound|Earthy,Sage,Spicy,Herbal|
|1024|Spicy,Herbal,Sage Woody|
       
# Impact
An ideal algorithm would be able to serve multiple purposes. It would allow new users with minimal knowledge of marijuana to enter what they believe would be desirable qualities in a strain and receive a recommendation for possible strains in the market based on historic user- aggregated information as well as strain content matching. Additionally, returning members could also receive new recommendations based on past reviews and ratings that user already entered into the system. The impact of this ideal model is two fold. From the consumers’ perspective, finding strains that are best suited to one’s liking means consumers can be as confident as possible that they’re getting the right product for them. This holds for both medicinal and recreational users. From the producers’ perspective, making users aware of products that they might enjoy could lead to more participation in the cannabis market and thus more revenue for dispensaries.

Beyond what has been described in this project, there are certainly other critical factors that may affect a user’s choice in marijuana. This could include variables such as the number of years the user has been smoking, the age of the user, etc. Data for these variables was difficult to gather data due to the site’s user confidentiality. Additionally, the extracted was only a small percentage of the large database Leafly had to offer. Due to the sparsity of the Incomplete matrix using all scrapable data from the site, the collaborative filtering approach did not yield strong results. With more diverse (and likely also just more) user data, the collaborative filtering approach might have achieved its goal. The data also suggested that there is only one archetypal cannabis smoker, yielding a very low rank (1) matrix that performed poorly on the test set. These suggestions for future attempts at building a recommendation system are to identify a target population with enough variance in strain taste and gather data on this population in order to train a useful and robust model.
