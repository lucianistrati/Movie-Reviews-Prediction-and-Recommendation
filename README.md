# Movie-Reviews-Prediction-and-Recommendation

----- What to do -----

- You are going to train a model on a given test set, using whatever tools you feel like, and submit predictions on an unlabeled test set
- Your predictions will be scored using a service we have set up, and the score will be returned back to you. We will also keep a log of the scores you have got.
- You can submit at most 20 sets of predictions per day
- The predictions must be submitted in the same order as the documents in the test directory. 
  E.g. the first prediction must be for "reviews0.txt", the second for "reviews1.txt" and so forth.



----- About the data -----

- The data is a collection of movie reviews. Each movie review is a pure text
- The training examples are labeled with a "score", which is the values you are going to predict
- The "score" is a value from 1 to 10, which is the rating given for this review
- For each training sample the first row of the file is the label, the rest is the content of the review
- 10,000 training examples (with labels) are given, and 2,000 test examples (without labels) are given



----- What scoring function is used? -----

Mean absolute error (https://en.wikipedia.org/wiki/Mean_absolute_error)
--> A lower score is better

The scoring function acepts both integers and decimal numbers.



----- How to submit your predictions -----

--- If you have python installed ---
If you have python installed, we have created a module that you can use to submit the predictions called 'score_predictions.py'.
You can either import the 'score' method from this module or use the 'score_predictions_from_file' method from the command line.


Example 1:

from score_predictions import score
predictions = model.predict(X)
score = score(list(predictions))

Example 2:

1. Write predictions (separated by commas) to file, e.g. "1,10,3,5,2" -> predictions.csv
2. Execute the following in the command line: 

	python score_predictions.py predictions.csv


--- If you do not have python installed: ---
If you do not want to use python, you can submit from anything able to send an http-request. 
Instructions:

- Send POST requests to the API url you we're given
- The body of the request must be a JSON with one entry with the key 'predictions'. Example:

	{
	  "predictions": [1,10,3,5,2]
	}

- The request header must contain the header value "x-api-key". Example:


	{
	  "x-api-key": <the key you were given>
	}

