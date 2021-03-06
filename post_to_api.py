import requests
import json

if __name__=="__main__":

    below_50k = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
    }
    r = requests.post("https://mlops-c3-project.herokuapp.com/inference/", data=json.dumps(below_50k))
    # should return 'success'
    assert r.json() == {"inference_result": "<=50K"}
    assert r.status_code == 200
    print("Inference result: " + r.json()["inference_result"])
    print("Exit code: " + str(r.status_code))
