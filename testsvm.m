function test_accuracy = testsvm(test_data, labels, weight_vector, bias)
% Test linear SVM 
% Input:
%  test_data: M*D matrix, each row as a sample and each column as a
%  feature
%  labels: M*1 vector, each row as a label
%  weight_vector: feature vector 
%  bias: bias term

values = (weight_vector')*(double(test_data)') + bias;

y = zeros(length(values),1);
y(values>=0) = 1;
y(values<0) = -1;

test_accuracy = sum(y==labels)/length(labels);