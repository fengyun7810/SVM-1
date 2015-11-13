function[cross_valid_accuracy,avg_time,Optimal_C]=cross_validation_linearsvm(train_data,train_label)
%Select optimal C value using 3-fold cross validation
%
%Input:
%train_data=N*D matrix with N samples and D features
%train_label=N*1 vector with N labels
%
%Output:
%accuracy: avg accuracy over 3 validation sets
%train_time:avg over each training subset

C=[4^-6 4^-5 4^-4 4^-3 4^-2 4^-1 1 4^1 4^2];
%3-fold cross validation
K=3;
cross_valid_accuracy=zeros(length(C),1);
avg_time = zeros(length(C),1);

%For all values of C
for index=1:length(C)
    %k-fold validation split
    N=size(train_data,1);
    Indices = crossvalind('Kfold', N,3);
    
    average_validation_accuracy=0;
    time=0;
    for i=1:K
        
        set_true = Indices~=i;
        x=train_data(set_true,:);
        y=train_label(set_true,:);
        validation_x=train_data(~set_true,:);
        validation_y=train_label(~set_true,:);
        
        startTime = tic;
        [w,b]=trainsvm(x,y,C(index));
        time = time + toc(startTime);
        average_validation_accuracy =  average_validation_accuracy + testsvm(validation_x, validation_y, w, b);
    end
   cross_valid_accuracy(index)= average_validation_accuracy/K;
   avg_time(index)=time/K;
end

[value,idx] = max(cross_valid_accuracy);
Optimal_C = C(idx);
disp(['C = ',num2str(Optimal_C ),' Max Accuracy = ',num2str(value*100),' Execution Time = ',num2str(avg_time(idx))]);