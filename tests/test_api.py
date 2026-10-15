from fastapi.testclient import TestClient

from recsys.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_search_ltr():
    r = client.get("/search", params={"q": "ergonomic mesh desk chair", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 5
    assert sum(1 for x in body["results"] if x["category"] == "desk chair") >= 4


def test_search_validation():
    assert client.get("/search", params={"q": "x", "mode": "bogus"}).status_code == 422
