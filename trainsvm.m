function[weight_vector,bias]=trainsvm(train_data,labels,C)

[N,D]= size(train_data);
H = eye(D+1+N); %weight vectoe,bias,ksi
H(D+1:end,D+1:end) = 0; %quadratic is only for weight vector

f = C * ones(D+1+N,1);
f(1:D+1) = 0; % w+b will be 0

I=ones(N, 1);
A = repmat(labels,1,D+1) .* [train_data ones(N,1)];  % for y.x & y.b
A = -1 * [A eye(N)]; % adding eye(N) for ksi
A=1*double(A);
b = -1 *I;

%lower bound
lb = [-inf(D+1, 1);zeros(N, 1)];

opts = optimoptions('quadprog','Algorithm','interior-point-convex');
[result, fval]= quadprog(H, f, A, b, [], [], lb, [],[],opts);

weight_vector = result(1:D);
bias = result(D+1);