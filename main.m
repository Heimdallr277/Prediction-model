clear;
clc;
%% RF
input_train  = xlsread('data.xls','rockburst','B2:G179');   
output_train=  xlsread('data.xls','rockburst','I2:I179');   
Y=xlsread('data.xls','rockburst','I180:I250');   
% Data preprocessing
[mtrain,ntrain] = size(input_train);
[dataset_scale,ps] = mapminmax(input_train',0,1);% Normalization
input_train= dataset_scale';
model = classRF_train(input_train,output_train,300,4);
importance=model.importance;
importance=importance/sum(model.importance);
disp('Feature importance: ')
disp(importance);
%% AHP
A=input("Discriminant matrix: ");
[m,n]=size(A);
[V,D]=eig(A);
tempNum1=D(1,1);
pos=1;
for h=1:n
    if D(h,h)>tempNum1
        tempNum1=D(h,h);
        pos=h;
    end
end    
w=abs(V(:,pos));
w=w/sum(w);
t=D(pos,pos);
disp('Max eigenrrot t=');disp(t);
CI=(t-n)/(n-1);
RI=[0 0 0.52 0.89 1.12 1.26 1.36 1.41 1.46 1.49 1.52 1.54 1.56 1.58 1.59 1.60 1.61 1.615 1.62 1.63];
CR=CI/RI(n);
if CR<0.10
    disp('Success');
    disp('CI=');disp(CI);
    disp('CR=');disp(CR);
else
    disp('Fail');
end

%% CM
load parameter.mat
%EX
EX(:,1)= parameter(:,2);
EX(:,2)=(parameter(:,2)+parameter(:,3))/2;
EX(:,3)=(parameter(:,3)+parameter(:,4))/2;
EX(:,4)=(parameter(:,4)+parameter(:,5))/2;
%EN
EN(:,1)=parameter(:,2)/3;
EN(:,2)=(parameter(:,3)-parameter(:,2))/6;
EN(:,3)=(parameter(:,4)-parameter(:,3))/6;
EN(:,4)=(parameter(:,5)-parameter(:,4))/6;
He=0.01;
%% Membership
output=input("Predict data   ");
[N,~]=size(output);
for j=1:N
    for i=1:4
        
     Enn1 = randn(1).*He + EN(1,i);
     u1(j,i)=exp(-(output(j,1)-EX(1,i)).^2./(2.*Enn1.^2));  
        
     Enn2 = randn(1).*He + EN(2,i);
     u2(j,i)=exp(-(output(j,2)-EX(2,i)).^2./(2.*Enn2.^2)); 
        
     Enn3 = randn(1).*He + EN(3,i);
     u3(j,i)=exp(-(output(j,3)-EX(3,i)).^2./(2.*Enn3.^2));
        
     Enn4 = randn(1).*He + EN(4,i);
     u4(j,i)=exp(-(output(j,4)-EX(4,i)).^2./(2.*Enn4.^2)); 
        
     Enn5 = randn(1).*He + EN(5,i);
     u5(j,i)=exp(-(output(j,5)-EX(5,i)).^2./(2.*Enn5.^2)); 
        
     Enn6 = randn(1).*He + EN(6,i);
     u6(j,i)=exp(-(output(j,6)-EX(6,i)).^2./(2.*Enn6.^2));     
     end     
end
%% Weights
for j=1:N
    	u(j,1)=u1(j,1)*w(1)+u2(j,1)*w(2)+u3(j,1)*w(1)+u4(j,1)*w(4)+u5(j,1)*w(5)+u6(j,1)*w(6);
        u(j,2)=u1(j,2)*w(1)+u2(j,2)*w(2)+u3(j,2)*w(1)+u4(j,2)*w(4)+u5(j,2)*w(5)+u6(j,2)*w(6);
        u(j,3)=u1(j,3)*w(1)+u2(j,3)*w(2)+u3(j,3)*w(1)+u4(j,3)*w(4)+u5(j,3)*w(5)+u6(j,3)*w(6);
        u(j,4)=u1(j,4)*w(1)+u2(j,4)*w(2)+u3(j,4)*w(1)+u4(j,4)*w(4)+u5(j,4)*w(5)+u6(j,4)*w(6);
end
%% Level
[max_a,y]=max(u,[],2);
%% Pics
tacc=length(find(Y==y))/length(y);
figure;
confusionchart(output,Y);
xlabel('predicted value');
ylabel('true value');
figure;
hold on;
plot(Y,'bo-','linewidth',0.8, 'markersize',6 ,'markerfacecolor', 'b')
plot(y,'rs-','linewidth',0.8, 'markersize',6 ,'markerfacecolor', 'r')
legend('True data','Predicted data');
title(['Comparison of True data and prediction data','(Accuracy=',num2str(tacc*100),'%)'],'FontSize',12);
xlabel('Number of rockburst cases','FontSize',12);
ylabel('Class','FontSize',12);
yticks([1 2 3 4])
grid on;

figure;
error =abs(y- Y);
plot(error,'rd','markersize',6 ,'markerfacecolor', 'r');
title('Absolute error ','FontSize',12);
xlabel('Number of rockburst cases','FontSize',12);
ylabel('Error ','FontSize',12);
grid on;
yticks([0 1  2])
legend('Error');

figure;
error = abs((y - Y)./Y);
plot(error','bo-','markersize',6 ,'markerfacecolor', 'b');
title('Relative error ','FontSize',12);
xlabel('Number of rockburst cases','FontSize',12);
ylabel('Relative error ','FontSize',12);
grid on;