# Movie-Recommendation-
Movie Recommendation using KNN and NLP-TFID vectorization with deployment on streamlit


1. Introduction 
What are Recommendation systems?
The recommendation systems are important as they help them make the right choices, without having to expend their cognitive resources. The purpose of a recommendation system basically is to search for content that would be interesting to an individual. Moreover, it involves a number of factors to create personalised lists of useful and interesting content specific to each user/individual. 
Movie recommendation systems aim at helping movie enthusiasts by suggesting what movie to watch without having to go through the long process of choosing from a large set of movies which go up to thousands and millions that is time consuming and confusing. Our aim is to reduce the human effort by suggesting movies based on the user’s interests. To handle such problems, we introduced a model combining both content-based and collaborative approach. 
These results are based on their profile, search/browsing history, what other people with similar traits/demographics are watching, and how likely are you to watch those movies. This is achieved through predictive modelling and heuristics with the data available.

   1.1 Objective 
The purpose of the recommendation system is to ensure customer satisfaction by delivering the correct content and optimising the time spent by the user on your website or channel. This also helps to increase customer engagement.
Movie recommendation systems 
	helps the user to find movies of their interest
	they help the provider to deliver movies to the right user and showcase personalised content to each user
	Increases revenues for business by improving user engagement
   1.2 Problem formation
For a website that streams movies by listing them all together for users to search and watch without providing any recommendation system will make it less interactive and will leave tiresome browsing experience without no suggestions for users about what to watch. This decreases user engagement with the system.
Therefore, to provide a recommendation system can fix the problem. Online streaming services like Netflix, Amazon Prime Video, Hotstar and many others maintain a database of movies in terms of reviews and ratings which can help to recommend movies to user according to their previous watch/search history, preferred genre of movies and other factors. This could increase their content consumption.  

Some of the examples of the pioneers in creating algorithms for recommendation systems and using them to serve their customers better in a personalized manner. These are:

GroupLens: 
– Developed initial recommender systems by collaborative filtering model 
Amazon:
– Implemented commercial recommender systems and a lot of computational improvements
Netflix:
– Latent Factor/ Matrix Factorization models
Google-Youtube:
– Hybrid Recommendation Systems
– Deep Learning based systems
– Social Network Recommendations

2. Literature survey 

Popularity based recommendation system
The popularity-based recommendation system eliminates the need for knowing factors like user browsing history, user preferences, the star cast of the movie, genre, and other factors. Hence, the single-most factor considered is the movie rating to generate a scalable recommendation system. 

Demerits of the popularity-based recommendation system
	Recommendations are not personalized as per user attributes and all users see the same recommendations irrespective of their preferences.
	The number of reviews will vary for each movie and hence the average star rating will have discrepancies. 
	The system doesn’t take into account the regional and language preferences and might recommend movies in languages that a regional dialect speaking individual might not understand


Collaborative Filtering 
One can measure the similarity between two users in different ways. A simple way would be to apply Pearson’s correlation to the common items. If the result is positively and highly correlated then the movies watched and liked by user A can be recommended to user B and vice-versa. On the other hand, if the correlation is negative then there is nothing to be recommended as the two users are not alike. 
Limitations of user-user collaborative filtering 
A user might be watching a specific niche type of movies that nobody else is watching. Hence there are no similar profiles resulting in no recommendations.
In case of a new movie, there are not enough user ratings to match 
In the case of a new user, there are not many movies that the user has watched or rated. Hence, it is difficult to map these users to similar users. 

Comparision table of work done by some researchers
Paper Name	Algorithm	Method & results 
	Movie
Recommendation System
By: - S. Rajarajeswari, Sharat Naik, Shagun Srikant, M. K. Sai Prakash and Prarthana Uday	Using the support vector machine algorithm and cosine similarity	The proposed hybrid system takes features from both collaborative as well as contentbased recommendation to come up with a system, which is more personalized for users.
It makes use of cosine similarity concept to find similar movies and the SVD prediction to estimate the movie recommendation as per the user’s taste.

	Content-Based Movie Recommendation
     System Using      Genre Correlation
SRS Reddy, Sravani Nalluri, Subramanyam Kunisetti, S. Ashok
	The approach used for building the recommendation system is content-based filtering. Use of dot product to calculate similarity 	Use of dot product to calculate similarity and use Euclidian distance to calculate the distance between the movies vector.
Retain the rows which have the minimum distance. These are the recommended
movies for the current user.
	Movie Recommendation System: Hybrid Information Filtering System Kartik Narendra Jain, Vikrant Kumar, Praveen Kumar and Tanupriya Choudhury	Coefficient Correlation and the content-based filter is the proposed algorithm that utilises the genres associated with the active users’ profile and those associated with the movies.	the movie genres are used to create clusters for clustering on items of the utility matrix the results of which are stored in matrix ‘clustered utility matrix’
After the creation of the similarity matrix, the system proceeds to the testing part and employs guessing of rating of test movies given by users on a particular movie by incorporating a set of similar users
A Novel Movie Recommendation System Based on Collaborative Filtering and Neural Networks Chu-Hsing Lin(&) and Hsuan Chi	
KNN algorithm and neural network <br>
Use the training data to calculate the similarity between each user.  <br>
Pick the most similar K neighbors for each user.  <br>
This experiment sets K to 35. Average the rating of each movie by the 35 selected neighbors as the predicted score.  <br>
Calculate the Mean Absolute Error (MAE) between the predicted score and the test data  <br>
Neural network has better accuracy and performance.  <br>
Movie Recommendation System Using Social
Network Analysis and k-Nearest Neighbor
Khamphaphone Xinchang, Doo-Soon	KNN algorithm 	To evaluate the recommendation accuracy of the algorithm, the most commonly used
measurement Mean Absolute Error (MAE) in CF recommender systems is used. MAE
can measure the prediction accuracy by calculating the deviation between user prediction
and real data. The smaller MAE is the higher accuracy of the recommendation.
MAE




3 Functionality of each component. 

	# KNN algorithm

Import the libraries which are Pandas, NumPy which are used for mathematical operation on the dataset using data frames.  <br>
We import the libraries using the command: -
import “library name” as “something”  <br>
So, for example: -  <br>
import pandas as pd  <br>
Thus, later on if we want to use the library, we do it using the keyword that we defined  <br>

	Import the datasets as data frames into the python file for analysis and model building.  <br>
The datasets are imported using the csv file with specification in using certain columns like the movie Id and title from the movies.csv file and userId , movieId and the rating column from the ratings.csv file.  <br>

	 Since in both the datasets we have movie Id common, we merge the datasets using that column  <br>

	Movie rating count function determines which user has given how much rating to each movie. We first drop the NaN columns and then we group by title on basis on rating. The rating is renamed to total ratings count.  <br>

	The recommendation is based on rating and hence we need total ratings since we cannot judge it on basis on a single user. The feature considered is title.  <br>

	On using the describe() function on the data frame, we get the count, mean, 25%, 50% and 75% quartiles and the min and max values.  <br>

	The popularity threshold is taken as 50 on analysis. So, the movies having rating count higher than 50 will be recommended.  <br>

	Pivot matrix is created with userId and title. The user which has given the rating for the movie will be shown and otherwise it is filled with 0. This matrix is important for cosine similarity.  <br>

	This matrix is converted into array matrix which is done using scipy.sparse library.  <br>

	Here the algorithm called K Nearest Neighbour is used because we use cosine similarity. This is unsupervised machine learning algorithm. We set the metric as cosine.  <br>

	We choose a random number now to run the model on this movie selected. We get 10 neighbors. We get two parameters: - distance and indices.  <br>

	We run a loop around the distances, if i==0; we print the statement recommendation for the movies. Then we print the movie name with distance parameter on the screen. The indices.flatten()  given the movie name. The movies are displayed on screen with increasing order of distance.  <br>


	# TFID vectorization  <br>

	Importing the datasets using dataframes  <br>

	The model works on genres tag from where it imports sentence  <br>

	The tf-idf weight is calculated which means uniqueness of a word is calculated.  <br>

	Using fit.transform(), we convert the corpus to matrix of word vectors.  <br>

	Generate cosine similarity matrix. The cosine score is calculated between tf-idf vectors.  <br>

	For cosine similarity, the value in NLP falls in the range of (0,1)  <br>

	We use linear kernel instead of cosine similarity because of smaller execution time.  <br>

	get_recommendations() function that takes in the title of a movie, a similarity matrix and an indices series as its arguments and outputs a list of most similar movies.  <br>

	Both the models are deployed on streamlit on local host. The file is run using the command: -  <br>
Streamlit run filename.py


4. Methodology and concepts
 
	Pandas makes importing, analysing, and visualizing data much easier. It builds on packages like NumPy and matplotlib to give you a single, convenient, place to do most of your data analysis and visualization work.  <br>

We use df.head() in case we need to print the first 5 rows of the dataset for verification where df is the data frame.
describe() on the dataframe is used for getting the mathematical results like mean, standard deviation and the quartiles which seem helpful for further calculations.
shape() shows the columns and the rows contained in the dataset
Pivot matrix is used for displaying the user id and the rating given by each user to the movie shown in the table.  <br>

	Cosine similarity:
Each movie has a vector of rating. On plotting, we can find out the Euclidian distance between the 2 vectors i.e. the 2 movies. But in case of that if we sue cosine similarity which means that if we find the angle between the vectors, we can get the similarity since cosine 90 is 0 and cosine 0 is 1 which will indicate maximum similarity. At cosine 45, we get 0.53 which means the movies have a similarity of 53%.  <br>

	KNN algorithm  <br>
K-nearest neighbours (KNN) algorithm uses ‘feature similarity’ to predict the values of new datapoints which further means that the new data point will be assigned a value based on how closely it matches the points in the training set.
After importing the dataset, we need to choose the value of K i.e. the nearest data points. K can be any integer.
 For each point in the test data do the following −
	 Calculate the distance between test data and each row of training data with the help of any of the method namely: Euclidean, Manhattan or Hamming distance. The most commonly used method to calculate distance is Euclidean.
	 Now, based on the distance value, sort them in ascending order.
	 Next, it will choose the top K rows from the sorted array.
	 Now, it will assign a class to the test point based on most frequent class of these rows.  <br>

	TFID vectorization  <br>

TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency.
Term Frequency (TF): The number of times a word appears in a document divided by the total number of words in the document. Every document has its own term frequency.
Inverse Data Frequency (IDF): The log of the number of documents divided by the number of documents that contain the word w. Inverse data frequency determines the weight of rare words across all documents in the corpus.
Lastly, the TF-IDF is simply the TF multiplied by IDF.
TF-IDF score of term I in document j  =TF(i,j)*IDF(i)  <br>

This is very common algorithm to transform text into a meaningful representation of numbers which is used to fit machine algorithm for prediction.
In TfidfVectorizer we consider overall document weightage of a word. It helps us in dealing with most frequent words. TfidfVectorizer weights the word counts by a measure of how often they appear in the documents.  <br>

	Deployment on local host using streamlit  <br>
Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.
We can run apps from the prompt using the command 
Streamlit run project.py  <br>
There are 2 parts to the web app i.e. the sidebar and the main page.  <br>

