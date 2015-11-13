function[coefficient,deg,maxAccuracy,accuracy,time]=nonlinear_libsvm_poly(train_data,train_label)
%
%Input:
%train_data=N*D matrix with N samples and D features
%train_label=N*1 vector with N labels
%
%Output:
%accuracy: avg accuracy over 3 validation sets
%train_time:avg over each training subset
disp('----------------');
disp('----------------');
disp(' ');
disp('Polynomial Kernel');
C=[4^-6 4^-5 4^-4 4^-3 4^-2 4^-1 1 4^1 4^2];
degree=[1 2 3];

accuracy=zeros(length(C),length(degree));
time = zeros(length(C),length(degree));

%For all values of C
for index=1:length(C)   
    for deg=1:length(degree)
        options = ['-q -s 0 -t 1 -v 3 -c',' ',num2str(C(index)),' ','-d',' ',num2str(degree(deg))];
        time_start = tic;
        accu = svmtrain(double(train_label),double(train_data),options);
        endTime = toc(time_start);
        accuracy(index,deg) = accu;
        time(index,deg) = endTime;
    end
end
disp('Polynomial Kernel');
for i=1:length(C)
    for j=1:length(degree)
    disp(['C = ',num2str(C(i)),' degree = ',num2str(degree(j)),' Accuracy = ',num2str(accuracy(i,j)),' Time = ' , num2str(time(i,j))]); 
    end
end

[maxAccuracy,I] = max(accuracy(:));
[n,m] = ind2sub(size(accuracy),I);

coefficient = C(n);
deg = degree(m);

disp(' ');
disp(['C = ',num2str(coefficient),' Degree = ',num2str(deg),' Max Accuracy = ',num2str(maxAccuracy),' Time = ',num2str(time(n,m))]);
