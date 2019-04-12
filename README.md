# Spam Filtering

A Machine Learning model is implemented using Scikit-Learn. 
First a Dictionary is created with the words occuring in the text, along with their frequencies. 
Then specific words that do not have an impact on whether the text is spam or not are removed. 
Finally, the word count is extracted consisting of 3000 dimensions. 
The performance is evaluated using Support Vector Machines and Multinomials Naive Bayes models.
