The repo contains the work of Clément Guillo (ENSAE, ENS Paris Saclay) and Hugo Thimonier (ENSAE, ENS Paris Saclay) which focuses on applying Hierarchical Optimal Topic Transport (Yurochkin et al. 2019) to predict NYT comment categories using the Kaggle dataset (https://www.kaggle.com/aashita/nyt-comments).

Our work focuses on a subsample of the comments posted in April 2018 which allows to balance the dataset according to the categories we whish to predict. Our approach consists in first trying to predict the categories of the article considering the 31 categories. However, our knn classification using the HOTT metric performs porrly suggesting that comments are not different enough in terms of topics they address between categories. The t-sne representation also supports that statement as well as the confusion matrix which does not display any confusion pattern for any category.

<p align="center">
  <img width="329" height="319" src="https://github.com/hugothimonier/HOTT_NLP_ENSAE/blob/master/img/T_SNE_30.png">
</p>


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
