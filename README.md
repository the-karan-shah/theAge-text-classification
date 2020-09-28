# Text Classification Project

In this project, I have built a flask application that can classify newspaper articles into various categories. 
The underlying model is currently a Random Forest model that has been trained using web scraped articles from The Age newspaper. 
The functions for the scraping and text cleaning can be found in the preprocessing.py files. 
The model currently only uses TF-IDF as the training parameter, but I have plans to add in Word2Vec embeddings and cross validation to further improve the accuracy. 

