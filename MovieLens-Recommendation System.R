
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


library(ggplot2)
library(lubridate)

#Data Pre Processing
# Modify the year as a column in the both datasets
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
#Root Mean Square Error Loss Function
#Function that computes the RMSE for vectors of ratings and their corresponding predictors
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2,na.rm=T))
}

#Exploratory Data analysis
head(edx) 

#Summary statistics of edx dataset
summary(edx)

# Number of unique movies and users in the edx dataset 
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

#Movie ratings are in each of the following genres in the edx dataset
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

#Summary statistics of edx rating
summary(edx$rating)

#Movie that has the greatest number of ratings
edx %>% group_by(title)%>%summarise(number = n())%>%arrange(desc(number))

#The five most given ratings in order from most to least
head(sort(-table(edx$rating)),5)

#Plot of rating 
table(edx$rating)
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()
#From the above plot, half star ratings are less common than whole star ratings

#Average ratings of edx dataset
avg_ratings <- edx %>% group_by(year) %>% summarise(avg_rating = mean(rating)) 


#Data Analysis Strategies
##From the above EDA, we can model the User,Movie,Year effects

# Ratings Histogram
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "black") +
  xlab("Rating") +
  ylab("Count") +
  ggtitle("Ratings Histogram") +
  theme(plot.title = element_text(hjust = 0.5))     

## Table 20 movies rated only once
#These are noisy estimates which can increase our RMSE
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:20) %>%
  knitr::kable()
                                                                                                                                                                                                                                                
# The distribution of each users ratings for movies-User Effect
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users Bias")

# The distribution of movie biases as most of the blockbuster movies are highly rated-Movie Effect
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movie Bias")

#Estimating the trend of rating versus release year-Year Effect
edx %>% group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth()

#Model Creation
#Final Model
#Regularization using movies, users, years 
##Penalized least squares Approach [minimize an equation that adds a penalty]
# lambda is a tuning parameter. Using cross-validation to find optimal value of lambda
## Please note that the below code could take some time 
# b_i,b_u,b_y represent the movie,user,year effects respectively
lambdas <- seq(0, 5, 0.25)

rmses <- sapply(lambdas,function(l){
  
  #Calculate the mean of ratings from the edx training set
  mu <- mean(edx$rating)
  
  #Adjust mean by movie effect and penalize low number on ratings
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  #Adjust mean by user and movie effect and penalize low number of ratings
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #Adjust mean by user, movie,year effect and penalize low number of ratings
  b_y <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l), n_y = n())
  
  #Predict ratings in the training set to find optimal penalty value 'lambda'
  predicted_ratings <- edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(edx$rating,predicted_ratings))
})

plot(lambdas, rmses) #Plot lamdas vs rmses


lambda <- lambdas[which.min(rmses)]  #To find optimal lambda
lambda

#Apply lambda on Validation set 
#Derive the mean from the training set
mu <- mean(edx$rating)
#Calculate movie effect with optimal lambda
movie_effect_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())
#Calculate user effect with optimal lambda
user_effect_reg <- edx %>% 
  left_join(movie_effect_reg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())
#Calculate year effect with optimal lambda
year_reg_avgs <- edx %>%
  left_join(movie_effect_reg, by='movieId') %>%
  left_join(user_effect_reg, by='userId') %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+lambda), n_y = n())
#Predict ratings on validation set
predicted_ratings <- validation %>% 
  left_join(movie_effect_reg, by='movieId') %>%
  left_join(user_effect_reg, by='userId') %>%
  left_join(year_reg_avgs, by = 'year') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>% 
  .$pred
model_rmse <- RMSE(validation$rating,predicted_ratings)

#Result
rmse_results <- data_frame(method="Reg Movie, User, Year Effect Model",  
                                     RMSE = model_rmse)
rmse_results %>% knitr::kable()


