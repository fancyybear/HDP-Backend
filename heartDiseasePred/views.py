from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


# import matplotlib.pyplot as plt
# from io import StringIO
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import warnings
# from sklearn.model_selection import KFold,cross_val_score
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# import os
# from django.conf import settings
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from django.conf import settings

@csrf_exempt
def home(request):
  age = request.GET.get('age', 0)  # Default to 0 if missing
 
  try:
    age = int(request.GET.get('age', 0))
    gender = int(request.GET.get('gender', 0))
    cp = int(request.GET.get('cp', 0))
    chol = int(request.GET.get('chol', 0))
    trestbps = int(request.GET.get('trestbps', 0))
    fbs = int(request.GET.get('fbs', 0))
    restcg = int(request.GET.get('restcg', 0))
    thalach = int(request.GET.get('thalach', 0))
    exang = int(request.GET.get('exang', 0))
    oldpeak = float(request.GET.get('oldpeak', 0))
    slop = int(request.GET.get('slop', 0))
    ca = int(request.GET.get('ca', 0))
    thal = int(request.GET.get('thal', 0))
  except ValueError:
    return JsonResponse({"error": "Invalid input format"}, status=400)



 
  gender=int(gender)
  cp=int(cp)
  chol=int(chol)
  trestbps=int(trestbps)
  fbs=int(fbs)
  restcg=int(restcg)
  thalach=int(thalach)
  exang=int(exang)
  oldpeak=float(oldpeak)
  slop=int(slop)
  ca=int(ca)
  thal=int(thal)


  csv_path = os.path.join(settings.BASE_DIR, 'data', 'heart_disease-data.csv')

  if not os.path.exists(csv_path):
    return JsonResponse({"error": "CSV file not found"}, status=500)
  
  df = pd.read_csv(csv_path)
  x=df.drop(columns='target',axis=1)
  y=df['target']
  x=df.iloc[:,:-1].values
  y=df.iloc[:,-1].values
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=23)
  dt=DecisionTreeClassifier()
  y_pred = dt.fit(x_train,y_train)
  # inp=(62,1,0,140,268,0,0,160,0,3.6,0,2,2)
  inp=(int(age),gender,cp,trestbps,chol,fbs,restcg,thalach,exang,oldpeak,slop,ca,thal)
  arr=np.array(inp)
  print(inp)
  pred=y_pred.predict(arr.reshape(1,-1))
  print(pred)
  value = pred.item()
  return JsonResponse({'result': value})