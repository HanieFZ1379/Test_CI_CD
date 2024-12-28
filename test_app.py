import pytest
from app import app


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


def test_predict(client):
    response = client.post('/predict', json={"features": [35, 1, 0, 120, 198, 0, 1, 130, 1, 1.6, 1, 0, 3]})

    print(response)
    assert response.status_code == 200
    assert "prediction" in response.get_json()
