[train_data,train_label,test_data,test_label]=data_preprocess('phishing-train.mat','phishing-test.mat');
[cross_valid_accuracy,avg_time,Optimal_C]=cross_validation_linearsvm(train_data,train_label);
[test_accuracy]=own_linear(train_data,train_label,test_data,test_label,Optimal_C);
compile_libsvm();
[accuracy,time]=experiment_libsvm(train_data,train_label);
[coefficient,deg,maxAccuracy,accuracy,time]=nonlinear_libsvm_poly(train_data,train_label);
[coefficient,deg,maxAccuracy,accuracy,time]=nonlinear_libsvm_rbf(train_data,train_label);
[test_accuracy]=libsvm(train_data,train_label,test_data,test_label);