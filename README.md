# Movie Reviews Prediction and Recommendation

## What to Do

In this project, you will train a model on a provided training set and submit predictions on an unlabeled test set. Your predictions will be scored using a service we have set up, and the score will be returned to you. Please note the following guidelines:

- You can submit up to 20 sets of predictions per day.
- Predictions must be submitted in the same order as the documents in the test directory. For example, the first prediction must be for "reviews0.txt", the second for "reviews1.txt", and so forth.

## About the Data

The dataset consists of a collection of movie reviews, where each review is represented as pure text. The training examples are labeled with a "score," which is the value you are going to predict. The "score" ranges from 1 to 10, representing the rating given for each review. The dataset includes 10,000 labeled training examples and 2,000 unlabeled test examples.

## Scoring Function

The scoring function used for evaluation is the Mean Absolute Error (MAE), where a lower score indicates better performance. The MAE measures the average absolute difference between the predicted and actual values.

For detailed information about the MAE, refer to: [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error)

## How to Submit Your Predictions

### Using Python

If you have Python installed, you can use the provided module called 'score_predictions.py' to submit your predictions. You have two options:

#### Option 1:

Import the 'score' method from the module and use it to submit your predictions. Example:

```python
from score_predictions import score
predictions = model.predict(X)
score = score(list(predictions))
```

#### Option 2:

Write your predictions to a file (separated by commas) and use the command line to execute the scoring script. Example:

1. Write predictions to file, e.g., "1,10,3,5,2" -> predictions.csv
2. Execute the following command:

```bash
python score_predictions.py predictions.csv
```

### Without Python

If you do not have Python installed or prefer not to use it, you can submit predictions using any tool capable of sending an HTTP request. Follow these instructions:

- Send POST requests to the API URL provided to you.
- The request body must be a JSON object with one entry, with the key 'predictions'. Example:

```json
{
  "predictions": [1,10,3,5,2]
}
```

- Include the "x-api-key" header in the request with the value provided to you. Example:

```json
{
  "x-api-key": "<your-api-key>"
}
```

Feel free to reach out if you have any questions or need further assistance!

[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)
