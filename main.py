pip install --upgrade pip

!pip install surprise

!pip install prettytable==3.9.0

# Our code is completed using an online collaborative system. 
# Before running it, you will need to download the specified libraries. 
# If the run fails, please try reloading or copying to run on PyCharm/VSCode. Alternatively, you can watch our demo.

import sys
import pandas as pd
import requests
from tabulate import tabulate
import numpy as np
from ast import literal_eval
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from surprise import KNNBaseline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from prettytable import PrettyTable

movie_data = pd.read_csv('movies_info.csv')

class MovieRecommenderMenu:
    def __init__(self):
        self.rating_recommender = RatingRecommender(movie_data)
        self.keyword_recommender = KeywordRecommender(movie_data)
        self.content_recommender = ContentRecommender(movie_data)
        self.personal_knn_recommender = Personal_KNN_recommender()

    def display_menu(self):
        print("Welcome to the Movie Recommender System!")
        while True:
            print("\nMenu:\n1. Rating Recommender\n2. Keyword Recommender\n3. Content Recommender\n4. Personalized Recommender")
            try:
                choice = int(input("Enter your choice (1-4): "))
                self.process_choice(choice)
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 4.")

    def process_choice(self, choice):
        if choice == 1:
            n = int(input("Enter the number of movies to recommend: "))
            self.rating_recommender.recommend(n)
        elif choice == 2:
            title = input("Enter the title of the movie: ")
            result_table = self.keyword_recommender.recommend(title)
            print(f'Using the Keyword recommending method for the movie "{title}", the following are the top 10 related movies: ')
            print(result_table)
        elif choice == 3:
            title = input("Enter the title of the movie: ")
            result_table = self.content_recommender.recommend(title)
            print(f'Using the Content recommending method for the movie "{title}", the following are the top 10 related movies: ')
            print(result_table)
        elif choice == 4:
            try:
                user_id_input = int(input("Please enter your userID (1-610): "))
            except ValueError:
                print("Invalid input. Please enter a numeric userID.")
            else:
                result, movie_ids, scores = self.personal_knn_recommender.recommend(usrID=user_id_input, num=10)
                table = PrettyTable()
                table.field_names = ["Rank", "Title", "Similarity Score"]
                for i, (title, score) in enumerate(zip(result, scores), 1):
                    table.add_row([i, title, round(score, 6)])
                print("Using the Personalized recommending method, followings are the top 10 score movies:")
                print(table)
        
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

class RatingRecommender:
    def __init__(self, movie_data):
        self.movies = movie_data.copy()
        self.average_vote = self.movies['vote_average'].mean()
        self.average_count = self.movies['vote_count'].quantile(0.9)
        self.movies = self.movies.loc[self.movies['vote_count'] >= self.average_count]

    def weighted_rating(self, x):
        m = self.average_count
        c = self.average_vote
        v = x['vote_count']
        r = x['vote_average']
        return (v / (v + m) * r) + (m / (m + v) * c)

    def recommend(self, n):
        self.movies = self.movies.drop_duplicates(subset='id')  # Drop duplicates based on movie ID
        self.movies['score'] = self.movies.apply(self.weighted_rating, axis=1)
        self.movies = self.movies.sort_values('score', ascending=False)

        headers = ['Rank', 'Title', 'Vote Count', 'Vote Average', 'Score']
        data = [(rank, movie['title'], movie['vote_count'], movie['vote_average'], movie['score'])
                for rank, (index, movie) in enumerate(self.movies[['title', 'vote_count', 'vote_average', 'score']].head(n).iterrows(), start=1)]

        print('Using the Rating recommending method, followings are the top %d score movies: ' % n)
        print(tabulate(data, headers=headers, tablefmt='grid'))

class KeywordRecommender:
    def __init__(self, movie_data):
        self.movies = movie_data.copy()
        # Apply literal_eval to convert string representations to lists
        self.movies['keywords'] = self.movies['keywords'].apply(literal_eval)
        self.movies['genres'] = self.movies['genres'].apply(literal_eval)
        # Combine 'keywords' and 'genres' into a single string with consistent formatting
        self.movies['combined_features'] = self.movies.apply(self.combine_features, axis=1)

    def combine_features(self, row):
        combined = ' '.join(sorted(set(row['keywords'] + row['genres']), key=str.lower))
        return combined

    def recommend(self, title):
        try:
            # Use CountVectorizer to convert the combined features into a token count matrix
            count_vectorizer = CountVectorizer(stop_words='english')
            count_matrix = count_vectorizer.fit_transform(self.movies['combined_features'])

            # Calculate cosine similarity between movies based on the token count matrix
            cosine_sim = cosine_similarity(count_matrix, count_matrix)

            # Get the index of the movie in the dataset
            idx_matches = self.movies.index[self.movies['title'] == title].tolist()

            if not idx_matches:
                print(f"Movie '{title}' not found. Please enter a valid movie title.")
                return None

            idx = idx_matches[0]


            # Get the pairwise similarity scores of all movies with the input movie
            sim_scores = list(enumerate(cosine_sim[idx]))

            # Sort the movies based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the 10 most similar movies
            sim_scores = sim_scores[1:11]

            # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]

            result_df = self.movies.loc[movie_indices, ['title']]
            result_df['similarity_score'] = [i[1] for i in sim_scores]
            result_df = result_df.reset_index(drop=True)
            result_df.index = result_df.index + 1
            result_df.index.name = 'Rank'

            result_table = tabulate(result_df, headers='keys', tablefmt='grid')

            return result_table

        except KeyError:
            print(f"Movie '{title}' not found. Please enter a valid movie title.")
            return None

    

class ContentRecommender:
    def __init__(self, movie_data):
        self.movies = movie_data.copy()
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.movies['overview'] = self.movies['overview'].fillna('')
        tfidf_matrix = self.tfidf.fit_transform(self.movies['overview'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()

    def recommend(self, title):
        try:
            idx = self.indices[title]
        except KeyError:
            print(f"Movie '{title}' not found. Please enter a valid movie title.")
            return None

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]

        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]

        result_df = self.movies.loc[movie_indices, ['title']]
        result_df['similarity_score'] = similarity_scores
        result_df['Rank'] = range(1, len(result_df) + 1)

        # Reorder columns for better display
        result_df = result_df[['Rank', 'title', 'similarity_score']]

        # Convert the result to a tabulated string
        result_table = tabulate(result_df, headers='keys', tablefmt='grid', showindex=False)

        return result_table

class Personal_KNN_recommender:
    def __init__(self):
        # Load movie and rating data
        self.index = pd.read_csv('movies.csv')
        # Use the ‘Reader’ class from the ‘surprise’ library to process scoring data.
        self.reader = Reader()
        self.ratings = pd.read_csv('train.csv')
        self.testings = pd.read_csv('test.csv')
        
        # Prepare the data for Surprise
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], self.reader)
        trainset = data.build_full_trainset()

        # Define similarity options for kNN algorithm
        sim_options = {'name': 'pearson_baseline', 'user_based': True}

        # Initialize KNNBaseline algorithm
        self.algo = KNNBaseline(sim_options=sim_options)

        # train it with the training set
        self.algo.fit(trainset)

        # Create a list of unique user IDs
        self.userid = self.ratings['userId'].unique() 

    def get_similar_users(self, usrID, num=10):
        # Getting similar users for a given user ID
        user_inner_id = self.algo.trainset.to_inner_uid(usrID)
        # convert the user ID to an internal ID
        user_neighbors = self.algo.get_neighbors(user_inner_id, k=num)
        user_neighbors = [self.algo.trainset.to_raw_uid(inner_id) for inner_id in user_neighbors]
        return user_neighbors

    def debug(self):
        # A debug function for testing similar user retrieval functionality.
        similar_users = self.get_similar_users(1, 1)
        print(self.ratings[self.ratings.userId == 1].head())
        for i in similar_users:
            print(list(self.ratings[self.ratings.userId == i]['movieId']))

    def recommend(self, usrID, num=5):
        # It used to recommend movies for a specified user ID.
        # Gets the user's rated movies.
        existed_movie = list(self.ratings[self.ratings.userId==usrID]['movieId'])
        # Finds similar users.
        similar_users = self.get_similar_users(usrID, num)
        movies_dict = {}
        # Compiles and sorts movies rated by similar users, not yet seen by the user.
        for i in similar_users:
            movie = list(self.ratings[self.ratings.userId == i]['movieId'])
            vote = list(self.ratings[self.ratings.userId == i]['rating'])
            for j in range(len(vote)):
                if not (movie[j] in existed_movie):
                    if movie[j] in movies_dict.keys():
                        movies_dict[movie[j]] += vote[j]
                    else:
                        movies_dict[movie[j]] = vote[j]
        # Now include similarity scores in your results
        result = sorted(movies_dict.items(), key=lambda x: x[1], reverse=True)
        result = result[:num]
        recommending = []
        recommending_id = []
        recommending_score = []
        for i in result:
            recommending.append(self.index[self.index.movieId==i[0]]['title'].values[0])
            recommending_id.append(i[0])
            recommending_score.append(i[1])
        # Returns top 10 recommended movies with titles, IDs, and scores.
        return recommending, recommending_id, recommending_score

    def test(self, num = 10):
        # Testing the recommender system and writing results to a CSV file
        result = []
        for user in self.userid:
            _, ids = self.recommend(user, num)
            result.append(ids)
        with open("./result.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['userId', 'result'])
            for i,row in enumerate(result):
                writer.writerow([self.userid[i], row])




if __name__ == "__main__":
    recommender_menu = MovieRecommenderMenu()
    recommender_menu.display_menu()

    