# Load and preprocess a single image
img = tf.keras.preprocessing.image.load_img(
    'path/to/image', 
    target_size=(IMG_HEIGHT, IMG_WIDTH)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
img_array = img_array / 255.0

# Make prediction
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]