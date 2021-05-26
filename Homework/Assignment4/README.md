Assignment 4 - Network analysis

DESCRIPTION:                                                                                                                                    
This command-line tool will take a given dataset and perform simple network analysis. In particular, it will build networks based on entities appearing together in the same documents, like we did in class.

- Your script should be able to be run from the command line
- It should take any weighted edgelist as an input, providing that edgelist is saved as a CSV with the column headers "nodeA", "nodeB"
- For any given weighted edgelist given as an input, your script should be used to create a network visualization, which will be saved in a folder called viz.
- It should also create a data frame showing the degree, betweenness, and eigenvector centrality for each node. It should save this as a CSV in a folder called output.

METHODS:                                                                                                                                                                                                                                                                                 For this assignment a weighted edgelist is taken as input. Then a graph is created and plotted with networkx. The centrality measures are then calculated, including the eigenvector centrality, the betweenness centrality and the degree centrality. This is made into a dataframe and later saved as a csvfile. The plot of the network is also saved in a folder called "viz"                                                       


CLONING REPO AND INSTALLING DEPENDENCIES:                                                                                                    
                                                                                                                                             
To run scripts within this repository, I recommend cloning the repository and installing relevant dependencies in a virtual ennvironment:        
$ git clone https://github.com/aksroa/Language_Portfolio.git                                                                                           
$ cd Language_Portfolio                                                                                                                          
$ bash ./create_lang_venv.sh                                                                                                                                                                                                                                                 
                                                                                                                                                          
If some of the libraries is not installed properly you can install these manually by running the following in the terminal:                   

$ cd Language_Portfolio                                                                                                                                                   
$ source sentiment_environment/bin/activate                                                                                                  
$ pip install {module_name}                                                                                                                  
$ deactivate              


To run the script from the command linewith default parameters, type:                                                                                                    
$ cd Language_Portfolio                                                                                                                
$ source sentiment_environment/bin/activate                                                                                                 
$ cd Homework/Assignment3/src                                                                                                               
$ python network.py

If you wish to run it with your own parameters, you could specify these:

$ python network.py --edgelist {your_file.csv} --nodes {ex: 100} --save {ex: False}


DISCUSSION OF RESULTS:                                                                                                                       
The assignment seems to run fine, both from the notebook and from the terminal. The visualization of the network is created and saved and the centrality measures is also created and saved fine.
