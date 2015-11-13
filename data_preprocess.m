%preprocess the training and test data to make
%features with multiple values to features taking values only zero or one.
function[train_final_features,train_labels,test_final_features,test_labels]=data_preprocess(train_filename,test_filename)
train_data=load(train_filename);
train_features=train_data(1).features;
train_labels=transpose(train_data(1).label);
train_final_features=transform(train_features);

%%% load test data %%%
test_data=load(test_filename);
test_features=test_data(1).features;
test_labels=transpose(test_data(1).label);
test_final_features=transform(test_features);
%%% transfor train data %%%

function[final_features]=transform(features)
columns_tobe_transformed=[2 7 8 14 15 16 26 29];
column_values=[-1 0 1];
[rows,total_columns]=size(features);
final_features=[];
for col=1:total_columns
    %if column is to be transformed
    if(ismember(col,columns_tobe_transformed))
        % create columns and set values
        for val=1:length(column_values)
            new_feature_col=zeros(rows,1);
            find_val_rows=find(features(:,col)==column_values(val));
            new_feature_col(find_val_rows,1)=1;
            final_features=[final_features new_feature_col];
        end
   
    else
        final_features=[final_features features(:,col)];
   end
end