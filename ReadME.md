The repo contains the work of <b> Clément Guillo </b>(ENSAE, ENS Paris Saclay) and <b>Hugo Thimonier </b>(ENSAE, ENS Paris Saclay) which focuses on applying Hierarchical Optimal Topic Transport (Yurochkin et al. 2019) to predict NYT comment categories using the Kaggle dataset (https://www.kaggle.com/aashita/nyt-comments).

Our work focuses on a subsample of the comments posted in April 2018 balanced according to the categories we whish to predict. Our approach consists in first trying to predict the categories of the article considering the 31 categories. However, our knn classification using the HOTT metric performs porrly suggesting that comments are not different enough in terms of topics they address between categories. 

The t-sne following representation supports that statement as it shows how all categories are mixed in the topic space. 

<p align="center">
  <img src="https://github.com/hugothimonier/HOTT_NLP_ENSAE/blob/master/img/T_SNE_30.png">
</p>

Similarly, the confusion matrix does not display any confusion pattern for any category also suggesting no clear topic differences in the comments between categories.


<p align="center">
  <img width ='80%' height ='80%' src="https://github.com/hugothimonier/HOTT_NLP_ENSAE/blob/master/img/confusion_matrix_30.png">
</p>

When reducing the categories to only 7 (Dining,  Games,  Foreign,  ArtsLeisure,  Science,  Sportsand Climate), the model performances highly improve, displaying less than 40% of test error. This can be explained by high differences in terms of topics addressed in the comments as the following t-sne representation suggests. 

<p align="center">
  <img width = '80% height = '80%' src="https://github.com/hugothimonier/HOTT_NLP_ENSAE/blob/master/img/T_SNE_7.png">
</p>

The similiraties and differences that the t-sne representation suggests can also be found in the confusion matrix of our model, where we can observe clear confusion pattern : when Climate is not correctly predicted, the most recurrent error is to  predict Foreign, the converse is also true. Similarly,the most  correctly  predicted category is Science, which was the most isolated category on the graph.   Finally,  the Art category which was the least isolated category on the graph is  also  the  category  which  is  the  least  correctly predicted.

The repo contains :
<p> 
	<b> .py files </b> :
	<p> • data.py which contains the functions required to load dataset in the format required to perform our analysis </p>
	<p> • distances.py which contains the distances metric functions used in the papers we refer to (mostly wmd) </p>
	<p> • hott.py which contains the hott metric functions </p>
	<p> • knn_classifer.py which contains the functions that allow the knn classification using the homemade metrics </p>
</p>
<p> <b> File 'papier' </b> : contains the papers used to perform our analysis. </p>
<p> <b> File 'dataset' </b> : contains the reduced dataset used to perform our analysis. </p>
<p> <b> File 'notebook' </b> : contains the notebook to run the functions (could be replaced by .py later on) and the generated files. <p>

 <p>
	To run the file you need to download the Glove pretrained embedding (https://nlp.stanford.edu/projects/glove/). 
</p>
