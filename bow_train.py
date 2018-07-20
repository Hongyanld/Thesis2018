df = read_data(args.data_file)
labels_series = df['Language']
text_series = df['Text']
Reading index from toefl11_tokenized.tsv
feature_matrix = TfidfVectorizer().fit_transform(text_series).toarray()
encoder = LabelEncoder().fit(labels_series)
file_name = os.path.basename(sys.argv[0]).split('.')[0]
check_cb = callbacks.ModelCheckpoint('checkpoints/' + str(file_name) + str(os.getpid()) + '.hdf5',
                                     monitor='val_loss', verbose=0, save_best_only=True, mode='min')
earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
# history = LossHistory()
x_train, x_test, y_train, y_test = train_test_split(feature_matrix, labels_series, test_size=0.09, random_state=0)
clf = LinearSVC(dual=False)
clf.fit(x_train, y_train)
LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
predictions = clf.predict(x_test)
print(classification_report(y_test, predictions))
             precision    recall  f1-score   support
     Arabic       0.81      0.79      0.80       103
    Chinese       0.72      0.76      0.74        84
     French       0.78      0.79      0.79        91
     German       0.85      0.91      0.88       102
      Hindi       0.67      0.65      0.66       114
    Italian       0.84      0.88      0.86        99
   Japanese       0.76      0.75      0.76       105
     Korean       0.66      0.74      0.70        82
    Spanish       0.83      0.69      0.76        98
     Telugu       0.69      0.76      0.72       112
    Turkish       0.83      0.68      0.74        99
avg / total       0.77      0.76      0.76      1089
feature_matrix.shape
(12100, 58487)
reduced = TruncatedSVD(n_components=500).fit_transform(feature_matrix)