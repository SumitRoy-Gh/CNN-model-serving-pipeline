from locust import HttpUser, task, between
import base64
import os

# Use an existing sample image from the workspace for load testing
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "Pizza image.jpg")

with open(IMAGE_PATH, "rb") as f:
    IMAGE_BYTES = f.read()

class FoodClassifierUser(HttpUser):
    wait_time = between(0.01, 0.05)   # simulate realistic request spacing

    @task
    def predict(self):
        # We pass the raw image bytes in the multipart/form-data request
        self.client.post(
            "/predict",
            files={"file": ("pizza.jpg", IMAGE_BYTES, "image/jpeg")}
        )
