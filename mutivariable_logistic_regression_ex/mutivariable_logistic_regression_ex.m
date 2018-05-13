clc;close all; clear all;
% get train
load 'train.txt';

%重新组合的样本集便于画图
train_c =  [];


%找到所有标签
labels = unique(train(:,end))';

%存储不同标签的样本的数量
set_size  = [];
n = 1
for i = labels
	%遍历每个样本, r为i对应的样本行号
	[r,c] =  find(train(:,end)==i);
	set_size(n) =  length(r);
	train_c= [train_c;train(r,:)];
	n = n+1
end;

figure;
p1 = set_size(1);
p2 = set_size(1)+set_size(2);
p3 = sum(set_size);

range1=1:p1;
range2=p1+1:p2;
range3=p2+1:p3;

xlabel('feature 1');
ylabel('feature 2');

plot(train_c(range1,1),train_c(range1,2),'rx','markersize', 10,'linewidth',8);
hold on;
plot(train_c(range2,1),train_c(range2,2),'bx','markersize', 10,'linewidth',8);

hold on;
plot(train_c(range3,1),train_c(range3,2),'g*','markersize', 10,'linewidth',8);

%定义costFunction
function [jVal,gradi] = costFunction(theta,X,y)
	m = size(X,1);
	% without regulazation
	%jVal = -(1/(m)) * sum ((y'*log(sigmoid_e(X*theta))+(1-y)'* log(1 - sigmoid_e(X*theta))));
	%gradi = X' *  (sigmoid_e(X*theta) - y ) / m ; 

	% with regulazation
	lambda = 0.9;
	new_theta = [0; theta(2:end)];
	jVal = -(1/(m)) * sum ((y'*log(sigmoid_e(X*theta))+(1-y)'* log(1 - sigmoid_e(X*theta)))) +   (lambda/(2*m)) * theta(2:end)'* theta(2:end);
	gradi = X' *  (sigmoid_e(X*theta) - y ) / m + lambda /m * new_theta ; 
end


function [thetaVal, functionVal, exitFlag] = calTheta(feature,data)
	%遍历每个样本, r为i对应的样本行号
	[r1,c] =  find(data(:,3)==feature);
	[r2,c] =  find(data(:,3)!=feature);

	train_c_binary = data;

	train_c_binary(r1,3) = 1;
	train_c_binary(r2,3) = 0;
	train_c_binary = [ones(size(train_c_binary,1),1) train_c_binary];

	X = train_c_binary(:,1:3);
	y = train_c_binary(:,4);
	m = size(train_c_binary,1);
	
	initTheta  =  zeros(3,1);

	option = optimset('GradObj','on','MaxIter',400);
	[thetaVal,functionVal,exitFlag]=fminunc(@(t)costFunction(t,X,y),initTheta,option);
end

thetas = zeros(3,3);
n = 1;
%画出决策边界
xx=linspace(0,0.1:5);
for i=labels
	[t, f, flag] = calTheta(labels(n),train);
	labels(n)
	t
	f
	flag
	thetas(:,n) = t;
	yy = - thetas(2,n)/thetas(3,n) .* xx - thetas(1,n)/thetas(3,n);
	plot(xx,yy,'linewidth',2,'k-');
	n = n+1;
end
function [val,class] =  predicts(x, thetas)
	vals = sigmoid_e(thetas'*x);
	vals
	[val,class] =  max(vals)
end

%thetas = zeros(3,1);
%n = 1;
%[t, f, flag] = calTheta(1,train);
%t
%f
%flag
%thetas(:,1) = t;
%function [val,class] =  predicts(x, thetas)
	%vals = sigmoid_e(thetas'*x);
	%vals
	%if vals > 0.5 class = 1; end
	%if vals < 0.5 class = 0; end
%end



while(1)
	x = input("input a x:\n");
	x = [1, x]
	hold on;
	plot(x(2),x(3),'ko','linewidth',8,'markersize',10);

	[val,class] = predicts(x',thetas);

	x
	disp('class')
	class
end




	




