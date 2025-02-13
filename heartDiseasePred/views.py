import os
import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

@csrf_exempt
def home(request):
    # Reject GET requests and enforce POST method
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method. Use POST instead."}, status=400)

    # Retrieve input values and handle missing data
    try:
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        cp = request.POST.get('cp')
        chol = request.POST.get('chol')
        trestbps = request.POST.get('trestbps')
        fbs = request.POST.get('fbs')
        restcg = request.POST.get('restcg')
        thalach = request.POST.get('thalach')
        exang = request.POST.get('exang')
        oldpeak = request.POST.get('oldpeak')
        slop = request.POST.get('slop')
        ca = request.POST.get('ca')
        thal = request.POST.get('thal')

        # Check if any required value is missing
        if None in [age, gender, cp, chol, trestbps, fbs, restcg, thalach, exang, oldpeak, slop, ca, thal]:
            return JsonResponse({"error": "Missing one or more required parameters"}, status=400)

        # Convert values to appropriate types
        age = int(age)
        gender = int(gender)
        cp = int(cp)
        chol = int(chol)
        trestbps = int(trestbps)
        fbs = int(fbs)
        restcg = int(restcg)
        thalach = int(thalach)
        exang = int(exang)
        oldpeak = float(oldpeak)
        slop = int(slop)
        ca = int(ca)
        thal = int(thal)
    except ValueError:
        return JsonResponse({"error": "Invalid input format. Ensure all inputs are numbers."}, status=400)

    # Load dataset and train model
    csv_path = os.path.join(settings.BASE_DIR, 'data', 'heart_disease-data.csv')
    df = pd.read_csv(csv_path)
    x = df.drop(columns='target', axis=1)
    y = df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=23)
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)

    # Prepare input for prediction
    input_data = np.array([age, gender, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slop, ca, thal]).reshape(1, -1)
    prediction = dt.predict(input_data)

    return JsonResponse({'result': int(prediction[0])})
