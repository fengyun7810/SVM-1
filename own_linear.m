
function[test_accuracy]=own_linear(train_data,train_label,test_data,test_label,coefficient)
[w,b] = trainsvm(train_data,train_label,coefficient);        
test_accuracy = testsvm(test_data,test_label,w,b);
disp(['Test Accuracy using C = ',num2str(coefficient),' ==> ',test_accuracy]);