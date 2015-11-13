function[accuracy,time]=experiment_libsvm(train_data,train_label)
%
%Input:
%train_data=N*D matrix with N samples and D features
%train_label=N*1 vector with N labels
%
%Output:
%accuracy: avg accuracy over 3 validation sets
%train_time:avg over each training subset

C=[4^-6 4^-5 4^-4 4^-3 4^-2 4^-1 1 4^1 4^2];

accuracy=zeros(length(C),1);
time = zeros(length(C),1);

%For all values of C
for index=1:length(C)    
        options = ['-q -s 0 -t 0 -v 3 -c',' ',num2str(C(index))];
        startTime = tic;
        accu = svmtrain(double(train_label),double(train_data),options);
        endTime = toc(startTime);
        accuracy(index) = accu;
        time(index) = endTime;
end

[value,idx] = max(accuracy);
Optimal_C = C(idx);
disp(['C = ',num2str(Optimal_C ),' Max Accuracy = ',num2str(value),' Execution Time = ',num2str(time(idx))]);