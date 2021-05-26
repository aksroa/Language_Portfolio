Assignment 6 - Text classification using Deep Learning

DESCRIPTION

Text classification using Deep Learning

Winter is... hopefully over.
                                                                                                                                                                                                                                                                                          
In class this week, we've seen how deep learning models like CNNs can be used for text classification purposes. For your assignment this week, I want you to see how successfully you can use these kind of models to classify a specific kind of cultural data - scripts from the TV series Game of Thrones.



You can find the data here: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons



In particular, I want you to see how accurately you can model the relationship between each season and the lines spoken. That is to say - can you predict which season a line comes from? Or to phrase that another way, is dialogue a good predictor of season?



Start by making a baseline using a 'classical' ML solution such as CountVectorization + LogisticRegression and use this as a means of evaluating how well your model performs. Then you should try to come up with a solution which uses a DL model, such as the CNNs we went over in class.                                                                                                                                                                                                                                                                                    

METHODS:

In terms of methods this assignment runs a logistic regression classifier on the Game of thrones data and a deep learning model. The baseline  model uses the CountVectorization method and a logistic regression, while the deep learning model uses the convolutional neural network method. 

CLONING REPO AND INSTALLING DEPENDENCIES:                                                                                                    
                                                                                                                                             
To run scripts within this repository, I recommend cloning the repository and installing relevant dependencies in a virtual ennvironment:        
$ git clone https://github.com/aksroa/Language_Portfolio.git                                                                                                      
$ cd Final_language_portfolio                                                                                                                          
$ bash ./create_lang_venv.sh                                                                                                                                                                                                                                                 
In addition you will need the glove6b50dtxt. This can be downloaded here: https://www.kaggle.com/watts2/glove6b50dtxt                                                                                                                                                                                                                                                                                                                
If some of the libraries are not installed properly you can install these manually by running the following in the terminal:                   

$ cd Language_Portfolio                                                                                                                                                                                                                                            
$ source sentiment_environment/bin/activate                                                                                                  
$ pip install {module_name}                                                                                                                  
$ deactivate                                                                                                                                                                                                                                                                             
                                                                                                                                                                                  To run the script from the command line, type:
                                                                                                                                            
$ cd Language_Portfolio                                                                                                                                
$ source sentiment_environment/bin/activate                                                                                                  
$ cd Homework/Assignment6/src                                                                                                                
$ python Assignment6-baseline.py (for baseline model)                                                
             
$ python Assignment6-deeplearning.py (for deep-learning model)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     DISCUSSION OF RESULTS:

The results of the models can ve found in the "output" folder. The results indicate, that when running the scripts with the defined default parameters on the Game of thrones data, the baseline model gets to an weighted accuracy of 0.26, while the CNN script achieves an accuracy on the training data of 0.1201, and of 0.1291 on the testing data. This means the model isn't performing well and that it is sligthly underfitting. But just a little bit
                                                                                                                                             
