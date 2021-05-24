Self-designed project                                                                                                                          

Description:                                                                                                                                    
For the self-designes assignment a dataset consisting of online customer reviews on hotel experiences have been used. The purpose of the assignment is to calculate the sentiment of the reviews and look at what type of words is used in these type of reviews.

Methods:                                                                                                                                        
For the self-designed project I started by creating a histogram of the ratings using plotly express. I then created a loop which looked over all the reviews and calculated the sentiment score of all of them, with a batchsize of 10.000. This was then inserted to the predefined polarity dataframe. Furthermore the sentimentscores was the grouped by the rating of the original review and visualized in a plot. Furthermore the reviews with high and low sentimentsscores were filtrated into two different lists, where a wordcloud were created of each of them. 

Cloning repo and installing dependencies                                                                                                        

To run scripts within this repository, I recommend cloning the repository and installing relevant dependencies in a virtual environment:        
$ git clone https://github.com/aksroa/Final_language_portfolio.git                                                                                    
$ cd Final_language_portfolio                                                                                                                        
$ bash ./create_lang_venv.sh                                                                                                                    

If some of the libraries is not installed properly you can install these manually by running the following in the terminal:                        
$ cd Final_language_portfolio                                                                                                                        
$ source sentiment_environment/bin/activate                                                                                                    
$ pip install {module_name}                                                                                                                    
$ deactivate                                                                                                                                    

Discussion of results:                                                                                                                          The assignment have been completed successfully and both the sentiments and the wordclouds have been created. From the histograms of the ratings it can be seen that there is a superiority of high ratings with over 9000 reviews with 5-stars and over 6000 reviews with 4-stars. When grouping the rating and the sentimentscore together it thus shows that the lower stars have quite high values of sentiment, which may be a little bit strange. This is also supported by the two wordclouds where positive words like "good" and "perfect" occur in the wordcloud with low sentimentscores.