import numpy as np
from sklearn.ensemble.forest import ExtraTreesClassifier

train_file = '/home/jvujjini/Kaggle/ForestCoverTypePrediction/train.csv'
test_file = '/home/jvujjini/Kaggle/ForestCoverTypePrediction/test.csv'

train_data = np.loadtxt(train_file, np.float32, delimiter=',')
test_data = np.loadtxt(test_file, np.float32, delimiter=',')

#training_data, training_label, test_data, test_label = train_data[:15000,:-1], data[:15000,-1], data[15000:,:-1], data[15000:,-1]

train_X = train_data[:,:-1]
train_y = train_data[:,-1]
test_X = test_data

clf = ExtraTreesClassifier(n_estimators=2500)
print "Started Training..."
clf.fit(train_X, train_y)
print "Done Training"
print "Started Predicting..."
predict_label = clf.predict(test_X)

output_file = '/home/jvujjini/Kaggle/ForestCoverTypePrediction/output.csv'

with open(output_file, 'w') as thefile:
    print "File Opened..."
    for item in predict_label:
        thefile.write("%s\n" % item)

print "Success!"