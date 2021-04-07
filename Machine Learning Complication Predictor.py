# NOTE: Much of the following code has been taken from the website
# https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python

import os.path
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy
import matplotlib.pyplot as plt
import seaborn

if not os.path.exists("Surgical-deepnet.csv"):
    print("Please place Surgical-deepnet.csv in the same directory as this program and then re-run the file.")
else:
    # load data from CSV file
    print("Loading data from CSV file...")
    ColNames = ['bmi', 'Age', 'asa_status', 'baseline_cancer', 'baseline_charlson', 'baseline_cvd',
                'baseline_dementia', 'baseline_diabetes', 'baseline_digestive', 'baseline_osteoart', 'baseline_psych',
                'baseline_pulmonary', 'ahrq_ccs', 'ccsComplicationRate', 'ccsMort30Rate', 'complication_rsi', 'dow',
                'gender', 'hour', 'month', 'moonphase', 'mort30', 'mortality_rsi', 'race', 'complication',
                'complication_or_mort30']
    File = pandas.read_csv("Surgical-deepnet.csv", header=None, names=ColNames)
    File.head()

    print("Choose your inputs\n(1) Raw inputs\n(2) All inputs")
    while True:
        ChooseInput = input("Enter: ")
        try:
            ChooseInput = int(ChooseInput)
            if ChooseInput == 1:
                FeatureCols = ['bmi', 'Age', 'baseline_cancer', 'baseline_cvd', 'baseline_dementia',
                               'baseline_diabetes', 'baseline_digestive', 'baseline_osteoart', 'baseline_psych',
                               'baseline_pulmonary', 'gender', 'race', 'dow', 'hour', 'month', 'moonphase']
                break
            elif ChooseInput == 2:
                FeatureCols = ['bmi', 'Age', 'baseline_cancer', 'baseline_charlson', 'baseline_cvd',
                               'baseline_dementia', 'baseline_diabetes', 'baseline_digestive', 'baseline_osteoart',
                               'baseline_psych',
                               'baseline_pulmonary', 'ahrq_ccs', 'ccsComplicationRate', 'ccsMort30Rate',
                               'complication_rsi', 'dow',
                               'gender', 'hour', 'month', 'moonphase', 'mortality_rsi', 'race']
                break
            else:
                print("Invalid input.")
        except ValueError:
            print("Invalid input.")

    Features = File[FeatureCols]  # actual feature data
    TargetVariable = File.complication_or_mort30  # label (target variable)
    print("Done.\n")

    print("Training logistic model...")
    # splits data into training (90%) and testing (10%)
    TrainFeatures, TestFeatures, TrainLabels, TestLabels = train_test_split(Features, TargetVariable, test_size=0.1, random_state=1)
    Batches = 50  # splits data into equal pieces
    Epochs = 20  # runs data through the model 20 times
    ListTrainFeatures = numpy.array_split(TrainFeatures, Batches)  # splits the data into 100 batches
    ListTrainLabels = numpy.array_split(TrainLabels, Batches)
    LogReg = LogisticRegression(max_iter=700, warm_start=True, random_state=1)
    print("0 %")
    for a in range(Epochs):
        for b in range(Batches):
            LogReg.fit(ListTrainFeatures[b], ListTrainLabels[b])  # trains model
        print(round(((a + 1) / Epochs * 100), 1), "%")
    print("Done.")

    # displays training
    LabelPrediction = LogReg.predict(TestFeatures)
    print("Accuracy:", metrics.accuracy_score(TestLabels, LabelPrediction))
    print("Precision:", metrics.precision_score(TestLabels, LabelPrediction))
    print("Recall:", metrics.recall_score(TestLabels, LabelPrediction))

    ConfusionMatrix = metrics.confusion_matrix(TestLabels, LabelPrediction)
    Classes = ["No complication", "Complication"]  # have complication or not
    fig, ax = plt.subplots()
    tick_marks = numpy.arange(len(Classes))
    plt.xticks(tick_marks, Classes)
    plt.yticks(tick_marks, Classes)

    # create confusion matrix heatmap
    seaborn.heatmap(pandas.DataFrame(ConfusionMatrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual outcome')
    plt.xlabel('Predicted outcome')
    plt.show()

    # evaluate performance
    y_pred_proba = LogReg.predict_proba(TestFeatures)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(TestLabels, y_pred_proba)
    auc = metrics.roc_auc_score(TestLabels, y_pred_proba)
    plt.plot(fpr, tpr, label="Data, AUC Score=" + str(auc))
    plt.legend(loc=4)
    plt.show()
