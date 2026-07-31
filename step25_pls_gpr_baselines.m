paperRoot = fileparts(mfilename('fullpath'));
clear;
clc;
close all;

%% 1. Data
resultFolder = fullfile(paperRoot, 'mill', 'results');
outputFolder = fullfile(paperRoot, 'additional_validation_results');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

timeData = load(fullfile(resultFolder,"quality_aware_with_time.mat"));
modelData = load(fullfile(resultFolder,"time_model_results.mat"));
T = timeData.timeAwareTable;

y = T.VB;
caseID = T.CaseID;
caseList = unique(caseID);
numberOfCases = numel(caseList);
numberOfSamples = height(T);
elapsedTime = T.ElapsedTime;
saturationRatio = T.smcDC_SaturationRatio;

allVariableNames = string(T.Properties.VariableNames);
sensorVariableNames = allVariableNames(8:85);
smcDCMask = startsWith(sensorVariableNames,"smcDC_");
otherSensorNames = sensorVariableNames(~smcDCMask);
XProcess = T{:,{'DOC','Feed','Material'}};
XAllSensors = T{:,cellstr(sensorVariableNames)};
XOtherSensors = T{:,cellstr(otherSensorNames)};
XAllTime = [XProcess,XAllSensors,elapsedTime,elapsedTime.^2];
XQualityTime = [XProcess,XOtherSensors, ...
    elapsedTime,elapsedTime.^2];

%% 2. Nested LOCO PLS and training-only GPR
componentGrid = [2 5 10 15 20];
predPLSAll = nan(numberOfSamples,1);
predPLSQuality = nan(numberOfSamples,1);
predGPRAll = nan(numberOfSamples,1);
predGPRQuality = nan(numberOfSamples,1);
bestPLSAllComponents = nan(numberOfCases,1);
bestPLSQualityComponents = nan(numberOfCases,1);

fprintf("========== PLS and GPR grouped baselines ==========" + newline);
for caseIndex = 1:numberOfCases
    testCase = caseList(caseIndex);
    testMask = caseID == testCase;
    trainMask = ~testMask;
    fprintf("  Outer test Case %d/%d\n",testCase,numberOfCases);

    [predPLSAll(testMask),bestPLSAllComponents(caseIndex)] = ...
        groupedPLSPrediction(XAllTime,y,caseID,trainMask,testMask, ...
        componentGrid);
    [predPLSQuality(testMask),bestPLSQualityComponents(caseIndex)] = ...
        groupedPLSPrediction(XQualityTime,y,caseID,trainMask,testMask, ...
        componentGrid);

    rng(20260718+caseIndex,"twister");
    predGPRAll(testMask) = fitGPRPrediction( ...
        XAllTime(trainMask,:),y(trainMask),XAllTime(testMask,:));
    rng(20261718+caseIndex,"twister");
    predGPRQuality(testMask) = fitGPRPrediction( ...
        XQualityTime(trainMask,:),y(trainMask),XQualityTime(testMask,:));
end

predPrimary = modelData.predictionTable.PredTimeGated;
modelNames = ["PrimaryTimeGated";"PLSAllSensorsTime"; ...
    "PLSQualityAwareTime";"GPRAllSensorsTime"; ...
    "GPRQualityAwareTime"];
predictionMatrix = [predPrimary,predPLSAll,predPLSQuality, ...
    predGPRAll,predGPRQuality];

%% 3. Metrics and case-level inference
numberOfModels = numel(modelNames);
MAE = nan(numberOfModels,1);
RMSE = nan(numberOfModels,1);
R2 = nan(numberOfModels,1);
P95AbsoluteError = nan(numberOfModels,1);
MaximumAbsoluteError = nan(numberOfModels,1);
caseMAE = nan(numberOfCases,numberOfModels);

for modelIndex = 1:numberOfModels
    residual = predictionMatrix(:,modelIndex)-y;
    MAE(modelIndex) = mean(abs(residual));
    RMSE(modelIndex) = sqrt(mean(residual.^2));
    R2(modelIndex) = 1-sum(residual.^2)/sum((y-mean(y)).^2);
    P95AbsoluteError(modelIndex) = prctile(abs(residual),95);
    MaximumAbsoluteError(modelIndex) = max(abs(residual));
    for caseIndex = 1:numberOfCases
        mask = caseID == caseList(caseIndex);
        caseMAE(caseIndex,modelIndex) = mean(abs( ...
            predictionMatrix(mask,modelIndex)-y(mask)));
    end
end

overallResults = table(modelNames,MAE,RMSE,R2,P95AbsoluteError, ...
    MaximumAbsoluteError);
caseResults = table(caseList,caseMAE(:,1),caseMAE(:,2),caseMAE(:,3), ...
    caseMAE(:,4),caseMAE(:,5),'VariableNames',{'CaseID', ...
    'PrimaryTimeGated','PLSAllSensorsTime','PLSQualityAwareTime', ...
    'GPRAllSensorsTime','GPRQualityAwareTime'});

comparisonNames = modelNames(2:end);
rawP = nan(numel(comparisonNames),1);
signedRankStatistic = nan(numel(comparisonNames),1);
meanDifferenceFromPrimary = nan(numel(comparisonNames),1);
bootstrapLower = nan(numel(comparisonNames),1);
bootstrapUpper = nan(numel(comparisonNames),1);

for comparisonIndex = 1:numel(comparisonNames)
    difference = caseMAE(:,comparisonIndex+1)-caseMAE(:,1);
    meanDifferenceFromPrimary(comparisonIndex) = mean(difference);
    [rawP(comparisonIndex),~,stats] = signrank( ...
        caseMAE(:,comparisonIndex+1),caseMAE(:,1));
    signedRankStatistic(comparisonIndex) = stats.signedrank;
    [bootstrapLower(comparisonIndex),bootstrapUpper(comparisonIndex)] = ...
        clusterBootstrapCI(difference,20262718+comparisonIndex);
end
holmAdjustedP = holmAdjustment(rawP);

statisticalResults = table(comparisonNames, ...
    repmat(numberOfCases,numel(comparisonNames),1), ...
    meanDifferenceFromPrimary,bootstrapLower,bootstrapUpper,rawP, ...
    holmAdjustedP,signedRankStatistic, ...
    'VariableNames',{'Comparator','IndependentCaseN', ...
    'MeanCaseMAEDifferenceFromPrimary','BootstrapCILower', ...
    'BootstrapCIUpper','RawP','HolmAdjustedP','SignedRankStatistic'});

hyperparameterResults = table(caseList,bestPLSAllComponents, ...
    bestPLSQualityComponents);

%% 4. Figure
fig = figure("Color","w","Position",[100 100 1350 580]);
tiledlayout(1,2,"TileSpacing","compact","Padding","compact");

ax1 = nexttile;
bar(ax1,categorical(modelNames),[MAE RMSE],"grouped");
grid(ax1,"on");
ylabel(ax1,"Error");
legend(ax1,["MAE","RMSE"],"Location","northwest");
title(ax1,"Strict LOCO comparison with PLS and GPR");
ax1.TickLabelInterpreter = "none";
xtickangle(ax1,20);

ax2 = nexttile;
bar(ax2,caseList,caseMAE,"grouped");
grid(ax2,"on");
xlabel(ax2,"Held-out case");
ylabel(ax2,"MAE");
legend(ax2,modelNames,"Location","northwest","Interpreter","none");
title(ax2,"Case-level baseline comparison");

for axisHandle = [ax1 ax2]
    if ~isempty(axisHandle.Toolbar)
        axisHandle.Toolbar.Visible = "off";
    end
    disableDefaultInteractivity(axisHandle);
end
drawnow;

%% 5. Save
predictionTable = table(T.OriginalIndex,caseID,T.RunID,y, ...
    predPrimary,predPLSAll,predPLSQuality,predGPRAll,predGPRQuality, ...
    'VariableNames',{'OriginalIndex','CaseID','RunID','ActualVB', ...
    'PrimaryTimeGated','PLSAllSensorsTime','PLSQualityAwareTime', ...
    'GPRAllSensorsTime','GPRQualityAwareTime'});

writetable(overallResults,fullfile(outputFolder, ...
    "pls_gpr_overall.csv"));
writetable(caseResults,fullfile(outputFolder, ...
    "pls_gpr_case_metrics.csv"));
writetable(statisticalResults,fullfile(outputFolder, ...
    "pls_gpr_statistics.csv"));
writetable(hyperparameterResults,fullfile(outputFolder, ...
    "pls_gpr_hyperparameters.csv"));
writetable(predictionTable,fullfile(outputFolder, ...
    "pls_gpr_predictions.csv"));

save(fullfile(outputFolder,"pls_gpr_baseline_results.mat"), ...
    "overallResults","caseResults","statisticalResults", ...
    "hyperparameterResults","predictionTable","componentGrid");

exportgraphics(fig,fullfile(outputFolder, ...
    "pls_gpr_comparison.png"),"Resolution",300);
exportgraphics(fig,fullfile(outputFolder, ...
    "pls_gpr_comparison.pdf"),"ContentType","vector");

fprintf("\n========== PLS/GPR results ==========\n");
disp(overallResults);
fprintf("\nFiles saved to:\n%s\n",outputFolder);


function [testPrediction,bestComponents] = groupedPLSPrediction( ...
    X,y,caseID,outerTrainMask,outerTestMask,componentGrid)

trainingCases = unique(caseID(outerTrainMask));
validationMAE = nan(numel(componentGrid),1);

for componentIndex = 1:numel(componentGrid)
    innerPrediction = nan(size(y));
    for validationIndex = 1:numel(trainingCases)
        validationMask = outerTrainMask & ...
            caseID == trainingCases(validationIndex);
        trainingMask = outerTrainMask & ...
            caseID ~= trainingCases(validationIndex);
        innerPrediction(validationMask) = fitPLSPrediction( ...
            X(trainingMask,:),y(trainingMask),X(validationMask,:), ...
            componentGrid(componentIndex));
    end
    validationMAE(componentIndex) = mean(abs( ...
        innerPrediction(outerTrainMask)-y(outerTrainMask)));
end

[~,bestIndex] = min(validationMAE);
bestComponents = componentGrid(bestIndex);
testPrediction = fitPLSPrediction(X(outerTrainMask,:),y(outerTrainMask), ...
    X(outerTestMask,:),bestComponents);
end


function prediction = fitPLSPrediction(XTrain,yTrain,XTest,components)
featureMean = mean(XTrain,1);
featureStd = std(XTrain,0,1);
keepFeature = featureStd > 1e-12;
XTrain = XTrain(:,keepFeature);
XTest = XTest(:,keepFeature);
featureMean = featureMean(keepFeature);
featureStd = featureStd(keepFeature);
XTrain = (XTrain-featureMean)./featureStd;
XTest = (XTest-featureMean)./featureStd;
components = min([components,size(XTrain,1)-1,size(XTrain,2)]);
[~,~,~,~,beta] = plsregress(XTrain,yTrain,components);
prediction = [ones(size(XTest,1),1),XTest]*beta;
end


function prediction = fitGPRPrediction(XTrain,yTrain,XTest)
featureStd = std(XTrain,0,1);
keepFeature = featureStd > 1e-12;
XTrain = XTrain(:,keepFeature);
XTest = XTest(:,keepFeature);
model = fitrgp(XTrain,yTrain, ...
    "BasisFunction","constant", ...
    "KernelFunction","matern32", ...
    "FitMethod","exact", ...
    "PredictMethod","exact", ...
    "Standardize",true);
prediction = predict(model,XTest);
end


function [lower,upper] = clusterBootstrapCI(difference,seed)
numberOfCases = numel(difference);
stream = RandStream("mt19937ar","Seed",seed);
bootstrapMean = nan(10000,1);
for bootstrapIndex = 1:10000
    sampledIndex = randi(stream,numberOfCases,numberOfCases,1);
    bootstrapMean(bootstrapIndex) = mean(difference(sampledIndex));
end
lower = prctile(bootstrapMean,2.5);
upper = prctile(bootstrapMean,97.5);
end


function adjustedP = holmAdjustment(rawP)
numberOfTests = numel(rawP);
[sortedP,order] = sort(rawP);
adjustedSorted = nan(numberOfTests,1);
runningMaximum = 0;
for testIndex = 1:numberOfTests
    currentAdjusted = (numberOfTests-testIndex+1)*sortedP(testIndex);
    runningMaximum = max(runningMaximum,currentAdjusted);
    adjustedSorted(testIndex) = min(1,runningMaximum);
end
adjustedP = nan(numberOfTests,1);
adjustedP(order) = adjustedSorted;
end
