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

def test_inference_correct_body_below50k():
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
    r = client.post("/inference/", json = below_50k)
    #should return 'success'
    assert r.json() == {"inference_result":"<=50K"}
    assert r.status_code == 200

def test_inference_correct_body_above50k():
    above_50k = {
                "age": 58,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 220531,
                "education": "Prof-school",
                "education-num": 15,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 100,
                "native-country": "United-States"
            }
    r = client.post("/inference/", json = above_50k)
    #should return 'success'
    assert r.json() == {"inference_result":">50K"}
    assert r.status_code == 200