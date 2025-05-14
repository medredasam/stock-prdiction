import tensorflow as tf

# Charger le modèle .keras
model = tf.keras.models.load_model('final_model2.keras')

# Sauvegarder au format .h5
model.save('final_model2.h5')

print("Conversion terminée !")
