function[coefficient,deg,maxAccuracy,accuracy,time]=nonlinear_libsvm_rbf(train_data,train_label)
%
%Input:
%train_data=N*D matrix with N samples and D features
%train_label=N*1 vector with N labels
%
%Output:
%accuracy: avg accuracy over 3 validation sets
%train_time:avg over each training subset
%RBF  Kernel

disp('----------------');
disp('----------------');
disp(' ');
disp('RBF Kernel');

C=[4^-6 4^-5 4^-4 4^-3 4^-2 4^-1 1 4^1 4^2];
gamma=[4^-7 4^-6 4^-5 4^-4 4^-3 4^-2 4^-1];

accuracy=zeros(length(C),length(gamma));
time = zeros(length(C),length(gamma));

%For all values of C
for index=1:length(C)   
    for ga=1:length(gamma)
        options = ['-q -s 0 -t 2 -v 3 -c',' ',num2str(C(index)),' ','-g',' ',num2str(gamma(ga))];
        time_start = tic;
        accu = svmtrain(double(train_label),double(train_data),options);
        endTime = toc(time_start);
        accuracy(index,ga) = accu;
        time(index,ga) = endTime;
    end
end
disp('RBF Kernel');
for i=1:length(C)
    for j=1:length(gamma)
    disp(['C = ',num2str(C(i)),' gamma = ',num2str(gamma(j)),' Accuracy = ',num2str(accuracy(i,j)),' Time = ' , num2str(time(i,j))]); 
    end
end

[maxAccuracy,I] = max(accuracy(:));
[n,m] = ind2sub(size(accuracy),I);

coefficient = C(n);
deg = gamma(m);

disp(' ');
disp(['C = ',num2str(coefficient),' Gamma = ',num2str(deg),' Max Accuracy = ',num2str(maxAccuracy),' Time = ',num2str(time(n,m))]);
