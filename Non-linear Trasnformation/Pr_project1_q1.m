clear
clc
close all

N = 2000;
h = 0;k = 0;a=12;b = 0;r = 2.5;
th = (1:N)*2*pi/N;
th = th(:);
xunit = h + a*cos(th);
yunit = k + b*sin(th);
plot(xunit,yunit);
figure;
 
%class 1
%mu1 is [h+a*cos(th),k+b*sin(th)] 
sigma1 = [15 0;0 1];
x = [randn(N,1)+h+a*cos(th),randn(N,1)+k+b*sin(th)];
plot(x(:,1),x(:,2),'bo','MarkerFaceColor','w');
hold on;

%class 2
%mu2 is [0,+/-7.5]
sigma2 = [1 0 ;0 1];
y1 = [0 + randn(N/2,1),7.5 + randn(N/2,1)];
y2 = [0 + randn(N/2,1),-7.5 + randn(N/2,1)];
y = [y1;y2];
plot(y(:,1),y(:,2),'r+','MarkerFaceColor','w');
legend('class 1','class 2');

%nonlinear transformation
lx1 = (sum(abs(x-[0,7.5])'));
lx2 = (sum(abs(x-[0,-7.5])'));
lx = abs(lx2 - lx1);
ly1 = (sum(abs(y-[0,7.5])'));
ly2 = (sum(abs(y-[0,-7.5])'));
ly = abs(ly1 - ly2);
figure;
plot(lx,'bs','LineWidth',1.5,'MarkerFaceColor','W');
hold on;
plot(ly,'ro','LineWidth',1.5,'MarkerFaceColor','W');
hold on;
xlabel('Distance');
legend('class 1','class 2');

%model
Z1(1:2:2*N-1,:) = x;
Z1(2:2:2*N,:) = y;
 TestTarg1(1:2:2*N-1) = 1;
 TestTarg1(2:2:2*N) = -1;
 vx = var(x);
 vy = var(y);
T = TestTarg1;
%PhiZ1 = (sum(abs(Z1-[0,7.5])'));
%PhiZ2 = (sum(abs(Z1-[0,-7.5])'));
%PhiZ = abs(PhiZ1 - PhiZ2);
PhiZ = [lx;ly];
Xmat = [ones(size(Z1,1),1) PhiZ(:)];
W_ls = regress(T(:),Xmat);
Y_x = Xmat*W_ls;
thr = -W_ls(1)/W_ls(2);

%predicting labels
pred_labels = ones(size(T));
pred_labels(Y_x < 0) = 2;
T(T == -1) = 2;
figure;
 plot(lx,'bs','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
 hold on;
 plot(ly,'ro','LineWidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
plot(PhiZ(pred_labels ==1),'k+','LineWidth',1.5,'MarkerFaceColor','w');
plot(PhiZ(pred_labels ==2),'y*','LineWidth',1.5,'MarkerFaceColor','w');
plot(thr*ones(N),'r','LineWidth',2);
hold off;
 
%Confusion matrix
ConfMat = confusionmat(T,pred_labels);
disp(ConfMat);
acc = sum(diag(ConfMat))/sum(sum(ConfMat));
disp(acc);
%%
%---------------------------------------------------------------------
% Classification model in the original attribute space
%---------------------------------------------------------------------
%  phiz1 = |z1-0| + |z2-7.5| , phiz2 = |z1-0| + |z2-(-7.5)|
%  phiz = |phiz1 - phiz2| = ||z2 - 7.5| - |z2 + 7.5|| = thr 
%by solving ||z2 - 7.5| - |z2 + 7.5|| = thr ,we get z2 = +/-thr/2 
%so here z1 can be anything from the given equation since it is not there
%in the euation 
mu = mean(Z1);
%z1vec = min(min([x(1,:),y(1,:)])):0.01:max(max([x(1,:),y(1,:)]));
z1vec = -2*thr:0.01:2*thr;
ix = 1;
for zx = 1:length(z1vec)
 z1 = z1vec(zx); 
 z2 = thr/2;
 model(ix,:) = [z1,z2];
 ix = ix + 1;
 z2 = -thr/2;
 model(ix,:) = [z1,z2];
 ix = ix+1;
end
figure;
plot(x(:,1),x(:,2),'bs',y(:,1),y(:,2),'ro','LineWidth',1.5,'MarkerFaceColor','w');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1','Class 2');
hold on;
plot(model(:,1),model(:,2),'g.','LineWidth',2);



