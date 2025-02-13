from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

import pandas as pd

import matplotlib.pyplot as plt
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
from django.conf import settings


@csrf_exempt
def home(request):
  age = request.POST.get('age')  # Retrieve 'age' field
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