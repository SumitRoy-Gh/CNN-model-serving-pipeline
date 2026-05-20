<div align="center">
  
# 🍕 Food-101 Classifier API

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Try%20it%20Live-orange)](https://huggingface.co/spaces/Sumitroyhuggingface/food_classifier)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Gradio](https://img.shields.io/badge/Gradio-API-blue)](https://gradio.app/)

*An end-to-end Machine Learning pipeline and free public REST API for classifying 101 different types of food using a fine-tuned EfficientNetB0 model.*

</div>

---

## 🌟 Overview

This repository contains the complete pipeline for training, exporting, and serving a deep learning model capable of recognizing **101 different food categories** (based on the Food-101 dataset). 

We didn't just stop at training a model; we deployed it! The model is currently hosted on Hugging Face Spaces and automatically exposes a **Free REST API**. You can integrate this powerful classifier into your own applications with zero setup, no API keys, and absolutely no cost.

👉 **[Try the Live Web App Here!](https://huggingface.co/spaces/Sumitroyhuggingface/food_classifier)**

---

## 🚀 How to Use the Free API

The Hugging Face deployment provides a blazing-fast, serverless API powered by Gradio. You can easily plug this into any frontend, backend, or mobile app.

### 🐍 Python (Backend / Scripts)
Perfect for data pipelines, Discord bots, or Python backends.

```bash
pip install gradio_client
```

```python
from gradio_client import Client

# Connect to the free hosted API
client = Client("Sumitroyhuggingface/food_classifier")

# Get predictions
result = client.predict(
    image="https://images.unsplash.com/photo-1513104890138-7c749659a591", # URL or local file path
    api_name="/predict"
)

print(result)
# Output: {'pizza': 0.94, 'flatbread': 0.04, ...}
```

### 🌐 JavaScript / TypeScript (Node.js, React, Next.js)
Ideal for web applications and mobile apps (React Native).

```bash
npm install @gradio/client
```

```javascript
import { Client } from "@gradio/client";

async function classifyFood() {
    const client = await Client.connect("Sumitroyhuggingface/food_classifier");
    
    const result = await client.predict("/predict", {
        image: "https://upload.wikimedia.org/wikipedia/commons/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg",
    });
    
    console.log("Top prediction:", Object.keys(result.data)[0]);
}
classifyFood();
```

### 🖥️ cURL (Direct HTTP)
Call it from anywhere, including bash scripts or older languages.

```bash
curl -X POST https://sumitroyhuggingface-food-classifier.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["https://your-image-url.jpg"]}'
```

---

## 💡 Example Project Ideas

Want to build something cool with this API? Here are some inspiration ideas:

1. **Diet & Macro Tracker App** 🥗
   - *How it works:* Users snap a picture of their meal. Your app calls this API to identify the food (e.g., `grilled_salmon`), then fetches nutritional data (calories, protein) from a database and logs it to their daily diary.
2. **Restaurant Review Aggregator** 🍔
   - *How it works:* A Yelp-style clone where users upload photos. The API tags the photos automatically so users can filter reviews by "Pizza", "Sushi", or "Tacos" without the uploader needing to tag them manually.
3. **Smart Refrigerator Inventory** 🧊
   - *How it works:* A Raspberry Pi camera in a fridge takes a photo, the API classifies the food, and a companion app generates a shopping list or suggests recipes based on what's currently available.
4. **Discord "Food Rater" Bot** 🤖
   - *How it works:* Whenever someone posts a food picture in a Discord channel, the bot uses this API to guess what it is and gives a sassy rating out of 10.

---

## 🏗️ Architecture & Technologies

This pipeline is built for scale, from data ingestion to cloud deployment.

- **Model Architecture**: `EfficientNetB0` (Pre-trained on ImageNet)
- **Deep Learning Framework**: `TensorFlow 2.x` & `Keras 3`
- **Data Pipeline**: `tf.data` API (optimized for high-throughput batching and prefetching)
- **Serving & UI**: `Gradio`
- **Hosting**: Hugging Face Spaces (Dockerized Container)
- **Dataset**: `Food-101` (~101,000 images, 101 classes)

### Local Development / Running the Code

If you want to run or train the model locally:

1. **Clone the repo:**
   ```bash
   git clone https://github.com/SumitRoy-Gh/CNN-model-serving-pipeline.git
   cd CNN-model-serving-pipeline
   ```
2. **Install dependencies:**
   ```bash
   pip install tensorflow>=2.16.1 gradio Pillow numpy
   ```
3. **Run the local server:**
   ```bash
   python hf_space/app.py
   ```
   *The app will start at `http://localhost:7860`.*

---

## 📋 Supported Food Categories (101 Classes)

<details>
<summary>Click here to expand the full list of 101 foods the model can recognize!</summary>

apple_pie, baby_back_ribs, baklava, beef_carpaccio, beef_tartare, beet_salad, beignets, bibimbap, bread_pudding, breakfast_burrito, bruschetta, caesar_salad, cannoli, caprese_salad, carrot_cake, ceviche, cheesecake, cheese_plate, chicken_curry, chicken_quesadilla, chicken_wings, chocolate_cake, chocolate_mousse, churros, clam_chowder, club_sandwich, crab_cakes, creme_brulee, croque_madame, cup_cakes, deviled_eggs, donuts, dumplings, edamame, eggs_benedict, escargots, falafel, filet_mignon, fish_and_chips, foie_gras, french_fries, french_onion_soup, french_toast, fried_calamari, fried_rice, frozen_yogurt, garlic_bread, gnocchi, greek_salad, grilled_cheese_sandwich, grilled_salmon, guacamole, gyoza, hamburger, hot_and_sour_soup, hot_dog, huevos_rancheros, hummus, ice_cream, lasagna, lobster_bisque, lobster_roll_sandwich, macaroni_and_cheese, macarons, miso_soup, mussels, nachos, omelette, onion_rings, oysters, pad_thai, paella, pancakes, panna_cotta, peking_duck, pho, pizza, pork_chop, poutine, prime_rib, pulled_pork_sandwich, ramen, ravioli, red_velvet_cake, risotto, samosa, sashimi, scallops, seaweed_salad, shrimp_and_grits, spaghetti_bolognese, spaghetti_carbonara, spring_rolls, steak, strawberry_shortcake, sushi, tacos, takoyaki, tiramisu, tuna_tartare, waffles
</details>

---

## 👨‍💻 Author

**Sumit Roy**
- GitHub: [@SumitRoy-Gh](https://github.com/SumitRoy-Gh)
- Hugging Face: [Sumitroyhuggingface](https://huggingface.co/Sumitroyhuggingface)

*Feel free to star ⭐ this repository if you find it helpful!*
