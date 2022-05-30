import requests
import os
import json
import sys


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
API_INFO_PATH = os.path.join(FILE_PATH, "apiinfo.json")


class APIException(Exception):
    pass


def _get_api_info():
    # Load the api key from file or get the api key from the user
    if not os.path.exists(API_INFO_PATH):
        api_url = input("Please input the API url: ").strip()
        api_key = input("Please input your given api key: ").strip()
        api_info = {"api_key": api_key, "api_url": api_url}
        with open(API_INFO_PATH, 'w') as f:
            f.write(json.dumps(api_info))
        print(f"Successfully saved api key to {API_INFO_PATH}. Please delete/edit this file if you gave the wrong key.")
    else:
        with open(API_INFO_PATH) as f:
            api_info = json.load(f)
    return api_info


def _get_api_url():
    # Load the api key from file or get the api key from the user
    if not os.path.exists(API_INFO_PATH):
        api_key = input("Please input your given api key: ").strip()
        with open(API_INFO_PATH, 'w') as f:
            f.write(api_key)
        print("Successfully saved api key to {}. Please delete/edit this file if you gave the wrong key.".format(API_INFO_PATH))
    else:
        with open(API_INFO_PATH) as f:
            api_key = f.read().strip()
    return api_key


def score(predictions):
    """
    Submit the given predictions to the API and score them

    Args:
        predictions (list): The predictions to submit.

    Example:
        score([1,2,3]) -> 0.12345

    Returns:
        float: The score.
    """

    # Check that predictions has correct type
    if not isinstance(predictions, list):
        raise TypeError("Predictions must be a list, not {}".format(type(predictions)))

    # Create the data object to send
    data = {
        "predictions": predictions
    }

    api_info = _get_api_info()

    # Assign the header values
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_info["api_key"]}

    # Submit the request
    response = requests.post(api_info["api_url"], data=json.dumps(data), headers=headers)

    try:
        return float(response.text)
    except:
        raise APIException(response.text)


def score_predictions_from_file(file_path):
    """
    Load predictions from a file and score them. The file containing the predictions is expected to contain
    a prediction for each sample separated by commas.

    Example of valid file content: "1,4,3,10,1,5"

    Args:
        file_path (str): The path to the file.

    Returns:
        float: The score.
    """
    with open(file_path, encoding='utf8') as f:
        predictions = [float(x) for x in f.read().strip().split(",")]
    return score(predictions)


if __name__ == '__main__':
    x = [5] * 2000
    print(score(x))

