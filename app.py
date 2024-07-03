from flask import Flask, render_template, request
import pandas as pd
from sklearn import model_selection
from sklearn import tree

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Separate features and target
X = data[['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']]
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize and train the model
model = tree.DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        ph = float(request.form['ph'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        
        new_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'ph': [ph],
            'humidity': [humidity],
            'rainfall': [rainfall]
        })
        
        predicted_crop = model.predict(new_data)[0]
        return render_template('predict.html', prediction=predicted_crop)
    return render_template('predict.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
