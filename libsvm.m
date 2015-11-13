function[test_accuracy]=libsvm(train_data,train_label,test_data,test_label)
%train data on polynomial and rbf kernel
%choose best among them

[coefficient_p,deg_p,maxAccuracy_p,accuracy_p,time_p]=nonlinear_libsvm_poly(train_data,train_label);
[coefficient_r,deg_r,maxAccuracy_r,accuracy_r,time_r]=nonlinear_libsvm_rbf(train_data,train_label);

if(maxAccuracy_p>maxAccuracy_r)
    options = ['-q -s 0 -t 1 -c',' ',num2str(coefficient_p),' ','-d',' ',num2str(deg_p)];
    model = svmtrain(double(train_label),double(train_data),options);
    result = svmpredict(double(test_label),double(test_data),model);
    test_accuracy = sum(result==test_label)/length(test_label);
    disp(['Polynomial Kernel: Test Accuracy using C = ',num2str(coefficient_p),' and degree = ',num2str(deg_p),' is ',num2str(test_accuracy*100)]);
else
    options = ['-q -s 0 -t 2 -c',' ',num2str(coefficient_r),' ','-g',' ',num2str(deg_r)];
    model = svmtrain(double(train_label),double(train_data),options);
    result = svmpredict(double(test_label),double(test_data),model);
    test_accuracy = sum(result==test_label)/length(test_label);
    disp(['RBF Kernel: Test Accuracy using C = ',num2str(coefficient_r),' and Gamma = ',num2str(deg_r),' is ',num2str(test_accuracy*100)]);
end