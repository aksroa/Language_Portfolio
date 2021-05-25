import pandas as pd
import plotly.express as px

# For the wordcloud
from wordcloud import WordCloud

import os
import spacy

# Remember we need to initialise spaCy
nlp = spacy.load("en_core_web_sm")

import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

import matplotlib.pyplot as plt

spacy_text_blob = SpacyTextBlob()
nlp.add_pipe(spacy_text_blob)
    
import argparse

def main(input):

    # I am making the filepath where the data is going to be saved.
    df_path = os.path.join("Output/sentiments5")

    # Reading the data
    data = pd.read_csv(input)
    
    #Showing a sample of he data
    print(data.head(10))
    
    # Creating a histogram of the ratings
    fig = px.histogram(data, x="Rating")
    fig.update_traces(marker_color="blue",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
    fig.update_layout(title_text='Hotel Rating')
    
    #Define an empty list of polarity
    polarity = []
    
    # Create for loop that loops over all the headlines and calculates the sentiment
    for doc in nlp.pipe(data["Review"], batch_size=10000):
    
    # For every sentence calculate sentiment by adding polarity
        for sentence in doc.sents:
            polarity_score = sentence._.sentiment.polarity
        
            # Adding the scores to polarity
            polarity.append(polarity_score)
            
            
            # Adding polarity to coloumn in the data frame
            polarity_values = pd.Series(polarity)

            #Pandas is used to insert polarity value to the dataset
            data.insert(loc=0, column="polarity", value=polarity_values)

            # Check the updated data frame
            print(data.head(10))

            # The data consists of 20.491 reviews
            print(len(data))

            # Save the data as a csv file
            data.to_csv(df_path)

            # Read the csv file
            sentiment_df = pd.read_csv(df_path)

            # I group the polarity scores by the rating
            sentiment_by_rating = sentiment_df.groupby("Rating")["polarity"].mean()

            # Check the grouped values
            sentiment_by_rating.head(10)

            # Plot the sentiment bu rating to get a look
            plt.plot(sentiment_by_rating)

            # Add title
            plt.title("Sentiment score by Rating")

            # Add label to x-axis
            plt.xlabel("Rating")

            # add label to y-axis
            plt.ylabel("Sentiment score")

            # Add a legend
            plt.legend(['Sentiment']);

            # Save the plot as sentimentscore_by_rating in the Output folder
            plt.savefig("Output/sentimentscore_by_rating1")
#
            # Filtrate the reviews which has a higher sentimentscore than 0.9, and add them to the list sentiments_high
            sentiments_high = data.loc[data.polarity > 0.9]

            # We can see that there is 231 reviews where the sentiment is above 0.9
            print(len(sentiments_high))

            #Take a look at the first 10
            sentiments_high.head(10)

            # Create a wordcloud which shows the most common words in reviews with sentimentscores over 0.9

            # Get the text from the "Review-column"
            text = str(sentiments_high.Review)

            # Create and generate a word cloud image:
            wordcloud = WordCloud().generate(text)

            # Display the generated image:
            plt.imshow(wordcloud, interpolation='bilinear')

            # There is no axes
            plt.axis("off")

            # Save the plot as wordcloud_high in the output-folder
            plt.savefig("Output/wordcloud_high1")
#
            # Filtrate the reviews which has a lower sentimentscore than 0.0, and add them to the list sentiments_low
            sentiments_low = data.loc[data.polarity < -0.0]

            # We can see that there is 2930 reviews where the sentimentscore is lower than 0.0
            print(len(sentiments_low))

            # Start
            text = str(sentiments_low.Review)

            # Create and generate a word cloud image:
            wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white").generate(text)

            # Display the generated image:
            plt.imshow(wordcloud, interpolation='bilinear')

            # There is no axes
            plt.axis("off")

            # Save the plot as wordcloud_low in the Output-folder
            plt.savefig("Output/wordcloud_low1")

# Define behaviour when called from command line            
if __name__=="__main__":
    
     # Initialize parser
    parser = argparse.ArgumentParser()

    # Add parser arguments
    parser.add_argument(
        "-i",
        "--input", 
        type = str,
        default = os.path.join(os.path.join("tripadvisor_hotel_reviews.csv")), # Default when not specifying a path
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "str containing path to input file")
    
    args = parser.parse_args()
 # Running the main function   
    main(args.input)
    