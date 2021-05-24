Assignment 1 - Basic scripting with Python

DESCRIPTION:

Basic scripting with Python

Using the corpus called 100-english-novels found on the cds-language GitHub repo, write a Python programme which does the following:

- Calculate the total word count for each novel
- Calculate the total number of unique words for each novel
- Save result as a single file consisting of three columns: filename, total_words, unique_words

METHODS:
For this assignment we were supposed to use the corpus caled 100-english novels as data. So the first thing that was done was defining the path to this data. After this I created a dataframe consisting of the columns filenme, total words and unique words. Furthermore a loop was created to look over all the novels. Here I isolated the filenames, calculated the number of words in the novels and the number of unique words using len(set()). These result were then assigned to the predefined dataframe and saved as a csv.                                                                                                                                                                                  

CLONING REPO AND INSTALLING DEPENDENCIES:

                                                                                                                                                
To run scripts within this repository, I recommend cloning the repository and installing relevant dependencies in a virtual ennvironment:    
$ git clone https://github.com/aksroa/Final_language_portfolio.git                                                                             
$ cd Final_language_portfolio                                                                                                                          
$ bash ./create_lang_venv.sh                                                                                                                                                                                                                                                 
 
 
The data used in this assignment can be found here: https://github.com/computationalstylistics/100_english_novels. To reproduce the results you should download this.                                                                                                                                                                                                                                                                  


If some of the libraries is not installed properly you can install these manually by running the following in the terminal:                   


$ cd Final_language_portfolio

$ source sentiment_environment/bin/activate

$ pip install {module_name}

$ deactivate



DISCUSSION OF RESULTS: 
In this assignment there is not to much to discuss about the results beyond that the purpose have been reached and a file consisting of the filename, total words and unique words have been saved.

