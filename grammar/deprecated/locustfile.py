from locust import HttpUser, task, between
import os
import random

class GrammarServiceUser(HttpUser):
    wait_time = between(0.1, 0.5)  # Reduced wait time for more frequent requests
    
    @task(3)
    def test_health_check(self):
        with self.client.get("/_ah/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data and "models_loaded" in data:
                        response.success()
                    else:
                        response.failure("Invalid health check response structure")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def test_detailed_health(self):
        with self.client.get("/detailed-health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data and "model_objects" in data:
                        response.success()
                    else:
                        response.failure("Invalid detailed health check response structure")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Unexpected status code: {response.status_code}") 