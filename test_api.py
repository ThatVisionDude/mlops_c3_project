from fastapi.testclient import TestClient
import json

from main import app

client = TestClient(app)

def test_welcome():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome_message": "Welcome, Dear User"}

def test_inference_wrong_body():
    r = client.post("/inference/", None)
    #should return 'unprocessable entity'
    assert r.status_code == 422
    r = client.post("/inference/", json = {'bad': 'dictionary'})
    #should return 'unprocessable entity'
    assert r.status_code == 422

def test_inference_correct_body():
    example_dict = {
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
    r = client.post("/inference/", json = example_dict)
    #should return 'success'
    print(r.json())
    assert r.status_code == 200