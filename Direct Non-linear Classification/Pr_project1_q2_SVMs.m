 clear
clc
close all

N = 1000;
h1 = 4;k = 2.5;a=2;b = 3.25;h2 = 2;
th = (1:N)*pi/N;
th = th(:);
xunit = h1 + a*cos(th);
yunit = b*sin(th);
plot(xunit,yunit);
hold on;
xunit = h2 + a*cos(th);
yunit = k-b*sin(th);
plot(xunit,yunit);
figure;
 
%class 1
x = [0.1*randn(N,1)+h1+a*cos(th),0.1*randn(N,1)+b*sin(th)];
plot(x(:,1),x(:,2),'bo','LineWidth',1.5,'MarkerFaceColor','w');
hold on;

%class 2
y = [0.1*randn(N,1)+h2+a*cos(th),0.1*randn(N,1)+k-b*sin(th)];
plot(y(:,1),y(:,2),'rs','LineWidth',1.5,'MarkerFaceColor','w');
legend('class 1','class 2');

t1 = ones(N,1);
t2 = -ones(N,1);
T = [t1;t2];
rin=randperm(length(T));
T=T(rin);

one_vec = ones(2*N,1);
xmat1 = [x;y];
xmat = xmat1(rin,:);
X = xmat;

trainingData = [X T];

predictors = trainingData(:,1:end-1);
response = trainingData(:,end); 

%%SVM with gaussian kernel
%{
classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 0.35, ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [-1; 1]);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2018b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 2 columns because this model was trained using 2 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3'});

predictorNames = {'column_1', 'column_2'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_3;
isCategoricalPredictor = [false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
%}

%%SVM with polynomial kernal of degree 1
 classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [-1; 1]);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2018b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 2 columns because this model was trained using 2 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3'});

predictorNames = {'column_1', 'column_2'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_3;
isCategoricalPredictor = [false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');




%{
Here for polynomial of order 2,3 we have 'polynomial' as 'KernalFunction'
and 'PolynomialOrder' as 2,3 respectively i.e instead of [] we have 2 or 3
respectively
Here we took fine Gaussian where KernakScale is 0.35
For Medium Gaussian KernalScale is 1.4 i.e KernalScale will vary
for coarse Gaussian it is 5.7
%}
 
pred_labels = partitionedModel.kfoldPredict;
figure;
plot(x(:,1),x(:,2),'bs','linewidth',1.5,'MArkerSize',10,'MarkerFaceColor','w');
hold on;
plot(y(:,1),y(:,2),'ro','linewidth',1.5,'MArkerSize',10,'MarkerFaceColor','w');
plot(X(pred_labels == 1,1),X(pred_labels == 1,2),'k+','linewidth',1.5,'MArkerSize',10,'MarkerFaceColor','w');
plot(X(pred_labels == -1,1),X(pred_labels == -1,2),'y*','linewidth',1.5,'MArkerSize',10,'MarkerFaceColor','w');
legend('True Class1','True Class2','Pred Class1','Pred Class2');
hold off;

pred_labels(pred_labels == -1) = 2;
TestTarg = T;
TestTarg(T == -1) = 2;
%ConfMat = ConfusionMatrix2(pred_labels,TestTarg,2);
ConfMat = confusionmat(TestTarg,pred_labels);
disp(ConfMat);
acc = sum(diag(ConfMat))/sum(sum(ConfMat));
disp(acc);

figure;
plot(x(:,1),x(:,2),'bs','linewidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');
hold on;
plot(y(:,1),y(:,2),'ro','linewidth',1.5,'MarkerSize',10,'MarkerFaceColor','w');

SVInx = trainedClassifier.ClassificationSVM.IsSupportVector;

plot(X(SVInx == 1,1),X(SVInx == 1,2),'kv','linewidth',1.5,'MarkerFaceColor','w');

x1range = min(min([x(:,1),y(:,1)])):.01:max(max([x(:,1),y(:,1)]));
x2range = min(min([x(:,2),y(:,2)])):.01:max(max([x(:,2),y(:,2)]));

[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];
predictedspecies = predict(trainedClassifier.ClassificationSVM,XGrid);
figure;
gscatter(xx1(:),xx2(:),predictedspecies,'wbr');
hold on;
plot(x(:,1),x(:,2),'mo','linewidth',1.5,'MarkerSize',4,'MarkerFaceColor','y');
hold on;
plot(y(:,1),y(:,2),'ks','linewidth',1.5,'MarkerSize',4,'MarkerFaceColor','k');
hold off;
legend off;




