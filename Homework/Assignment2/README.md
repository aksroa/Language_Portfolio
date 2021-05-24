Assignment 2- String processing with Python

DESCRIPTION:

String processing with Python

Using a text corpus found on the cds-language GitHub repo or a corpus of your own found on a site such as Kaggle, write a Python script which calculates collocates for a specific keyword.

- The script should take a directory of text files, a keyword, and a window size (number of words) as input parameters, and an output file called out/{filename}.csv
- These parameters can be defined in the script itself
- Find out how often each word collocates with the target across the corpus
- Use this to calculate mutual information between the target word and all collocates across the corpus
- Save result as a single file consisting of three columns: collocate, raw_frequency, MI
- BONUS CHALLENGE: Use argparse to take inputs from the command line as parameters

METHODS:
In this assignment I created a function which looped over all the texts from the corpus, splitting by whitespaces, removing punctations and turning everything to lowercase and returning the number of times each collocate appear. Then the collocates were extracted from the dictionary. O11 and O12 along with R1 is then calculated. The collocate, raw frequency and the MI is the added to the dataframe.                                                                                                                                                                              

CLONING REPO AND INSTALLING DEPENDENCIES:

                                                                                                                                                
To run scripts within this repository, I recommend cloning the repository and installing relevant dependencies in a virtual ennvironment:      
$ git clone https://github.com/aksroa/Final_language_portfolio.git                                                                                                                                                                                                                                       
$ cd Final_language_portfolio                                                                                                                                                                                                                                                                  
$ bash ./create_lang_venv.sh                                                                                                                                                                                                                                                 


The data used in this assignment can be found here: https://github.com/computationalstylistics/100_english_novels. To reproduce the results 
                                                                                                                                             you should download this.                                                                                                                                                                                                                                                                
                                                                                                                                             
If some of the libraries are not installed properly you can install these manually by running the following in the terminal:                                                                                                                                                              
$ cd Final_language_portfolio                                                                                                                      
$ source sentiment_environment/bin/activate                                                                                                  
$ pip install {module_name}                                                                                                                  
$ deactivate                                                                                                                                  

DISCUSSION OF RESULTS: 

The assignment have been completed successfully and the collocates, raw frequency and MI can be found in the Output-folder. The code also runs from the command line.

To run it from the command line type                                                                                                          
$ cd Final_language_portfolio                                                                                                                
$ source sentiment_environment/bin/activate                                                                                                  
$ cd Homework/Assignment2/src                                                                                                                
$ python Collocation.py                                                                                                                      


