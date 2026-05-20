import tensorflow as tf
import json
import os

# Get absolute paths to make script independent of CWD
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# ── Load your already-trained model ──────────────────────────────────────────
model_path = os.path.join(script_dir, 'efficientnet_food_finetuned.keras')
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully")
print("Output shape:", model.output_shape)   # should be (None, 101)

# ── Class names for Food-101 (hardcoded — no tfds needed) ────────────────────
CLASS_NAMES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari',
    'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad',
    'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger',
    'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream',
    'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
    'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings',
    'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
    'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

assert len(CLASS_NAMES) == 101, f"Expected 101 classes, got {len(CLASS_NAMES)}"
print(f"Class names loaded: {len(CLASS_NAMES)} classes")

# ── Save class names for the API ──────────────────────────────────────────────
api_dir = os.path.join(project_root, 'api')
os.makedirs(api_dir, exist_ok=True)
with open(os.path.join(api_dir, 'class_names.json'), 'w') as f:
    json.dump(CLASS_NAMES, f)
print("class_names.json saved to api/")

# ── Define serving signature (accepts raw image bytes) using Keras 3 ExportArchive ─
export_archive = tf.keras.export.ExportArchive()
export_archive.track(model)

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
def serve_image_fn(image_bytes):
    def decode_and_preprocess(img_bytes):
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        return img

    images = tf.map_fn(
        decode_and_preprocess,
        image_bytes,
        fn_output_signature=tf.float32
    )
    return model(images, training=False)

export_archive.add_endpoint(
    name='serving_default',
    fn=serve_image_fn
)

# ── Export SavedModel (Using atomic rename to avoid TF Serving polling race conditions) ─
models_dir = os.path.join(project_root, 'serving', 'models', 'efficientnet_food')
os.makedirs(models_dir, exist_ok=True)
versions = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
next_version = max(versions) + 1 if versions else 1

EXPORT_PATH = os.path.join(models_dir, str(next_version))
temp_export_path = os.path.join(models_dir, f"temp_{next_version}")

# Clean up any failed temp paths
import shutil
if os.path.exists(temp_export_path):
    shutil.rmtree(temp_export_path)

os.makedirs(temp_export_path, exist_ok=True)
export_archive.write_out(temp_export_path)

# Atomic rename to the final version directory
if os.path.exists(EXPORT_PATH):
    shutil.rmtree(EXPORT_PATH)
os.rename(temp_export_path, EXPORT_PATH)
print(f"\nSavedModel exported to: {EXPORT_PATH}")

# ── Verify the export ──────────────────────────────────────────────────────────
loaded = tf.saved_model.load(EXPORT_PATH)
print("Signatures found:", list(loaded.signatures.keys()))
# Should print: ['serving_default']

print("\nExport complete. Ready for TF Serving.")