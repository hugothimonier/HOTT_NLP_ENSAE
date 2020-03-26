The repo contains :
<p> 
	<b> .py files </b> :
	<p> • data.py which contains the functions required to load dataset in the format required to perform our analysis </p>
	<p> • distances.py which contains the distances metric functions used in the papers we refer to (mostly wmd) </p>
	<p> • hott.py which contains the hott metric functions </p>
	<p> • knn_classifer.py which contains the functions that allow the knn classification using the homemade metrics </p>
</p>
<p> <b> File 'papier' </b> : contains the papers used to perform our analysis </p>
<p> <b> File 'notebook' </b> : contains the notebook to run the functions (could be replaced by .py later on <p>

 <p>
	To run the file you need to download the Glove pretrained embedding (https://nlp.stanford.edu/projects/glove/). Morover the NYT dataset is also required (https://www.kaggle.com/aashita/nyt-comments).
</p>

<p>
	<b> Future work to be done </b> : 
	<p> (i) rework on the stopword removal function because it does not seem to work as the LDA topics seem to show -> leads to bad performance of the model </p>
	<p> (ii) contruct another hott functions that allows t-sne function to use it (see doc for the function requisits) </p>