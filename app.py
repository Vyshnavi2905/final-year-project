
import pandas as  pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from flask import Flask, render_template, request
import os

app = Flask(__name__)
app.config['upload folder']='uploads'


@app.route('/')
def home():
    return render_template('index.html')
global path

@app.route('/load data',methods=['POST','GET'])
def load_data():
    if request.method == 'POST':

        file = request.files['file']
        filetype = os.path.splitext(file.filename)[1]
        if filetype == '.csv':
            path = os.path.join(app.config['upload folder'], file.filename)
            file.save(path)
            print(path)
            return render_template('load data.html',msg = 'success')
        elif filetype != '.csv':
            return render_template('load data.html',msg = 'invalid')
        return render_template('load data.html')
    return render_template('load data.html')


@app.route('/view data',methods = ['POST','GET'])
def view_data():
    file = os.listdir(app.config['upload folder'])
    path = os.path.join(app.config['upload folder'],file[0])

    global df
    df = pd.read_csv(path)



    print(df)
    return render_template('view data.html',col_name =df.columns.values,row_val = list(df.values.tolist()))

@app.route('/model',methods = ['POST','GET'])
def model():
    if request.method == 'POST':
        global scores1,scores2,scores3,scores4,scores5,scores6
        global df
        filename = os.listdir(app.config['upload folder'])
        path = os.path.join(app.config['upload folder'],filename[0])
        df = pd.read_csv(path)
        global testsize
        # print('hdf')
        testsize =int(request.form['testing'])
        print(testsize)
        # print('hdf')
        global x_train,x_test,y_train,y_test
        testsize = testsize/100
        # print('hdf')
        print(df)

        df = df.dropna()
        X = df.iloc[:,:10]
        y = df.Output
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        x_train,x_test,y_train,y_test = train_test_split(X_res, y_res,test_size=testsize,random_state=10)
        # print('ddddddcf')
        model = int(request.form['selected'])
        if model == 1:
            lr = LogisticRegression()
            model1 = lr.fit(x_train,y_train)
            pred1 = model1.predict(x_test)
            # print('sdsj')
            scores1 = accuracy_score(y_test,pred1)
            # print('dsuf')

            return render_template('model.html',score = round(scores1,4),msg = 'accuracy',selected  = 'Logistic Regression')
        elif model == 2:
            rfc = RandomForestClassifier(random_state = 10,criterion = 'gini',ccp_alpha=0.17)
            model2 = rfc.fit(x_train[:500],y_train[:500])
            pred2 = model2.predict(x_test)
            scores2 =accuracy_score(y_test,pred2)
            return render_template('model.html',msg = 'accuracy',score = round(scores2,3),selected = 'RANDOM FOREST CLASSIFIER')
        elif model == 3:
            dt = DecisionTreeClassifier()
            model3 = dt.fit(x_train[:500],y_train[:500])
            pred3 = model3.predict(x_test)
            scores3 = accuracy_score(y_test,pred3)
            return render_template('model.html',msg = 'accuracy',score = round(scores3,3),selected = 'Decision Tree Classifier ')
        elif model == 4:
            xgb = XGBClassifier()
            model4 = xgb.fit(x_train[:500],y_train[:500])
            pred4 = model4.predict(x_test)
            scores4 = accuracy_score(y_test,pred4)
            return render_template('model.html',msg = 'accuracy',score = round(scores4,3),selected = 'XGBClassifier')
        elif model == 5:
            nb = GaussianNB()
            model5 = nb.fit(x_train,y_train)
            pred5 = model5.predict(x_test)
            scores5 = accuracy_score(y_test,pred5)
            return render_template('model.html',msg = 'accuracy',score = round(scores5,3),selected = 'GaussianNB')
        elif model == 6:
            from sklearn.neural_network import MLPClassifier
            ML = MLPClassifier()
            model5 = ML.fit(x_train,y_train)
            pred5 = model5.predict(x_test)
            scores6 = accuracy_score(y_test,pred5)
            return render_template('model.html',msg = 'accuracy',score = round(scores6,3),selected = 'MLPClassifier')


    return render_template('model.html')

@app.route('/prediction',methods = ['POST',"GET"])
def prediction():
    if request.method == 'POST':

        b = float(request.form['b'])
        c = float(request.form['c'])
        d = float(request.form['d'])
        e = float(request.form['e'])
        f = float(request.form['f'])
        g = float(request.form['g'])
        h = float(request.form['h'])
        i = float(request.form['i'])
        j = float(request.form['j'])
        k = float(request.form['k'])

        values = [[float(b),float(c),float(d),float(e),float(f),float(g),float(h),float(i),float(j),float(k)]]

        dtc = DecisionTreeClassifier()
        model = dtc.fit(x_train,y_train)

        pred = model.predict(values)
        print(pred)
        type(pred)
        if pred == [0]:
            msg = 'The Software is safe'
        elif pred == [1]:
            msg = 'The Software is unsafe'

        return render_template('prediction.html',msg =msg)
    return render_template('prediction.html')

@app.route("/graph",methods=['GET','POST'])
def graph():
    i = [scores1,scores2,scores3,scores4,scores5,scores6]
    return render_template('graph.html',i=i)

if __name__ == '__main__':
    app.run(debug=True)