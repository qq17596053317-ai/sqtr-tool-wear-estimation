paperRoot = fileparts(mfilename('fullpath'));
clear;
clc;
close all;

%% 1. Data
scriptFolder = fileparts(mfilename("fullpath"));
addpath(scriptFolder);

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

predProposed = modelData.predictionTable.PredTimeGated;

%% 2. Exploratory safety-weighted model with nested grouped selection
% This is a secondary safety-oriented analysis, not a replacement for the
% prespecified primary model. All weight, lambda and gate choices are made
% using outer-training cases only.
highWearTrainingThreshold = 0.40;
weightFactorGrid = [1 2 4];
lambdaGrid = logspace(-4,4,17);
gateThresholdGrid = [0.01 0.10 0.50];

predSafetyWeighted = nan(numberOfSamples,1);
selectedWeightFactor = nan(numberOfCases,1);
selectedLambdaAll = nan(numberOfCases,1);
selectedLambdaQuality = nan(numberOfCases,1);
selectedGateThreshold = nan(numberOfCases,1);
selectedValidationLoss = nan(numberOfCases,1);

fprintf("========== Nested safety-weighted ridge ==========" + newline);
for caseIndex = 1:numberOfCases
    testCase = caseList(caseIndex);
    testMask = caseID == testCase;
    trainMask = ~testMask;

    candidateLoss = nan(numel(weightFactorGrid),1);
    candidateLambdaAll = nan(numel(weightFactorGrid),1);
    candidateLambdaQuality = nan(numel(weightFactorGrid),1);
    candidateGate = nan(numel(weightFactorGrid),1);

    for weightIndex = 1:numel(weightFactorGrid)
        currentFactor = weightFactorGrid(weightIndex);
        observationWeight = ones(numberOfSamples,1);
        observationWeight(y >= highWearTrainingThreshold) = currentFactor;

        [candidateLambdaAll(weightIndex),innerPredAll] = ...
            selectWeightedBranch(XAllTime,y,caseID,trainMask, ...
            lambdaGrid,observationWeight,highWearTrainingThreshold);
        [candidateLambdaQuality(weightIndex),innerPredQuality] = ...
            selectWeightedBranch(XQualityTime,y,caseID,trainMask, ...
            lambdaGrid,observationWeight,highWearTrainingThreshold);

        gateLoss = nan(numel(gateThresholdGrid),1);
        for gateIndex = 1:numel(gateThresholdGrid)
            innerGated = innerPredAll;
            switchMask = trainMask & ...
                saturationRatio >= gateThresholdGrid(gateIndex);
            innerGated(switchMask) = innerPredQuality(switchMask);
            gateLoss(gateIndex) = balancedWearLoss( ...
                y(trainMask),innerGated(trainMask), ...
                highWearTrainingThreshold);
        end
        minimumGateLoss = min(gateLoss);
        tiedGate = find(gateLoss <= minimumGateLoss + ...
            1e-12*max(1,abs(minimumGateLoss)));
        [~,tiePosition] = max(gateThresholdGrid(tiedGate));
        bestGateIndex = tiedGate(tiePosition);
        candidateGate(weightIndex) = gateThresholdGrid(bestGateIndex);
        candidateLoss(weightIndex) = gateLoss(bestGateIndex);
    end

    minimumCandidateLoss = min(candidateLoss);
    tiedWeight = find(candidateLoss <= minimumCandidateLoss + ...
        1e-12*max(1,abs(minimumCandidateLoss)));
    [~,tiePosition] = min(weightFactorGrid(tiedWeight));
    bestWeightIndex = tiedWeight(tiePosition);

    selectedWeightFactor(caseIndex) = weightFactorGrid(bestWeightIndex);
    selectedLambdaAll(caseIndex) = candidateLambdaAll(bestWeightIndex);
    selectedLambdaQuality(caseIndex) = ...
        candidateLambdaQuality(bestWeightIndex);
    selectedGateThreshold(caseIndex) = candidateGate(bestWeightIndex);
    selectedValidationLoss(caseIndex) = candidateLoss(bestWeightIndex);

    finalWeight = ones(sum(trainMask),1);
    finalWeight(y(trainMask) >= highWearTrainingThreshold) = ...
        selectedWeightFactor(caseIndex);
    predAll = fitWeightedRidge(XAllTime(trainMask,:),y(trainMask), ...
        finalWeight,XAllTime(testMask,:),selectedLambdaAll(caseIndex));
    predQuality = fitWeightedRidge(XQualityTime(trainMask,:),y(trainMask), ...
        finalWeight,XQualityTime(testMask,:), ...
        selectedLambdaQuality(caseIndex));
    predGated = predAll;
    switchMask = saturationRatio(testMask) >= ...
        selectedGateThreshold(caseIndex);
    predGated(switchMask) = predQuality(switchMask);
    predSafetyWeighted(testMask) = predGated;

    fprintf("  Case %d: weight=%g, gate=%.0f%%\n",testCase, ...
        selectedWeightFactor(caseIndex), ...
        100*selectedGateThreshold(caseIndex));
end

%% 3. Regression and wear-band reporting
modelNames = ["PrimaryTimeGated";"SafetyWeightedTimeGated"];
predictionMatrix = [predProposed,predSafetyWeighted];
overallResults = regressionResultTable(y,predictionMatrix,modelNames);

wearBandNames = ["VB < 0.2";"0.2 <= VB < 0.4";"VB >= 0.4"];
wearBandMask = {y < 0.2,y >= 0.2 & y < 0.4,y >= 0.4};
bandRows = cell(numel(modelNames)*numel(wearBandNames),7);
rowIndex = 0;
for modelIndex = 1:numel(modelNames)
    for bandIndex = 1:numel(wearBandNames)
        rowIndex = rowIndex+1;
        mask = wearBandMask{bandIndex};
        residual = predictionMatrix(mask,modelIndex)-y(mask);
        bandRows(rowIndex,:) = {modelNames(modelIndex), ...
            wearBandNames(bandIndex),sum(mask),mean(abs(residual)), ...
            sqrt(mean(residual.^2)),mean(residual),median(residual)};
    end
end
wearBandResults = cell2table(bandRows,'VariableNames', ...
    {'Model','WearBand','RecordN','MAE','RMSE','MeanResidual', ...
    'MedianResidual'});
wearBandResults.Model = string(wearBandResults.Model);
wearBandResults.WearBand = string(wearBandResults.WearBand);

%% 4. Critical-wear alert metrics at two analytical thresholds
alertThresholds = [0.30 0.40];
classificationRows = cell(numel(modelNames)*numel(alertThresholds),13);
rowIndex = 0;
for modelIndex = 1:numel(modelNames)
    for thresholdIndex = 1:numel(alertThresholds)
        rowIndex = rowIndex + 1;
        threshold = alertThresholds(thresholdIndex);
        actualPositive = y >= threshold;
        predictedPositive = predictionMatrix(:,modelIndex) >= threshold;
        TP = sum(actualPositive & predictedPositive);
        TN = sum(~actualPositive & ~predictedPositive);
        FP = sum(~actualPositive & predictedPositive);
        FN = sum(actualPositive & ~predictedPositive);
        sensitivity = safeDivide(TP,TP+FN);
        specificity = safeDivide(TN,TN+FP);
        precision = safeDivide(TP,TP+FP);
        F1 = safeDivide(2*precision*sensitivity,precision+sensitivity);
        balancedAccuracy = mean([sensitivity specificity],"omitnan");
        classificationRows(rowIndex,:) = {modelNames(modelIndex), ...
            threshold,TP,TN,FP,FN,sensitivity,specificity,precision,F1, ...
            balancedAccuracy,sum(actualPositive),sum(~actualPositive)};
    end
end
classificationResults = cell2table(classificationRows, ...
    'VariableNames',{'Model','AlertThreshold','TP','TN','FP','FN', ...
    'Sensitivity','Specificity','Precision','F1','BalancedAccuracy', ...
    'PositiveRecordN','NegativeRecordN'});
classificationResults.Model = string(classificationResults.Model);

%% 5. Case-level paired inference
caseMAE = nan(numberOfCases,numel(modelNames));
caseHighWearMAE = nan(numberOfCases,numel(modelNames));
caseHighWearN = nan(numberOfCases,1);
for caseIndex = 1:numberOfCases
    caseMask = caseID == caseList(caseIndex);
    highMask = caseMask & y >= highWearTrainingThreshold;
    caseHighWearN(caseIndex) = sum(highMask);
    for modelIndex = 1:numel(modelNames)
        caseMAE(caseIndex,modelIndex) = mean(abs( ...
            predictionMatrix(caseMask,modelIndex)-y(caseMask)));
        if any(highMask)
            caseHighWearMAE(caseIndex,modelIndex) = mean(abs( ...
                predictionMatrix(highMask,modelIndex)-y(highMask)));
        end
    end
end

caseResults = table(caseList,caseHighWearN,caseMAE(:,1),caseMAE(:,2), ...
    caseHighWearMAE(:,1),caseHighWearMAE(:,2), ...
    'VariableNames',{'CaseID','HighWearRecordN','PrimaryMAE', ...
    'SafetyWeightedMAE','PrimaryHighWearMAE','SafetyWeightedHighWearMAE'});

overallDifference = caseMAE(:,2)-caseMAE(:,1);
validHighCase = isfinite(caseHighWearMAE(:,1)) & ...
    isfinite(caseHighWearMAE(:,2));
highDifference = caseHighWearMAE(validHighCase,2) - ...
    caseHighWearMAE(validHighCase,1);
[overallP,~,overallStats] = signrank(caseMAE(:,2),caseMAE(:,1));
[highWearP,~,highWearStats] = signrank( ...
    caseHighWearMAE(validHighCase,2),caseHighWearMAE(validHighCase,1));
[overallLower,overallUpper] = clusterBootstrapCI( ...
    overallDifference,20263718);
[highLower,highUpper] = clusterBootstrapCI(highDifference,20264718);

statisticalResults = table( ...
    ["Overall case MAE";"High-wear case MAE"], ...
    [numberOfCases;sum(validHighCase)], ...
    [mean(overallDifference);mean(highDifference)], ...
    [overallLower;highLower],[overallUpper;highUpper], ...
    [overallP;highWearP], ...
    [overallStats.signedrank;highWearStats.signedrank], ...
    'VariableNames',{'Endpoint','IndependentCaseN','MeanDifference', ...
    'BootstrapCILower','BootstrapCIUpper','WilcoxonP', ...
    'SignedRankStatistic'});

hyperparameterResults = table(caseList,selectedWeightFactor, ...
    selectedLambdaAll,selectedLambdaQuality,selectedGateThreshold, ...
    selectedValidationLoss);

%% 6. Figures
fig = figure("Color","w","Position",[100 100 1450 590]);
tiledlayout(1,2,"TileSpacing","compact","Padding","compact");
colors = [0.10 0.45 0.75;0.85 0.35 0.10];

ax1 = nexttile;
hold(ax1,"on");
for modelIndex = 1:numel(modelNames)
    scatter(ax1,y,predictionMatrix(:,modelIndex),38, ...
        colors(modelIndex,:),"filled","MarkerFaceAlpha",0.60);
end
limits = [min([y;predictionMatrix(:)]) max([y;predictionMatrix(:)])];
plot(ax1,limits,limits,"k--","LineWidth",1.2);
xline(ax1,highWearTrainingThreshold,"Color",[0.4 0.4 0.4], ...
    "LineStyle",":");
yline(ax1,highWearTrainingThreshold,"Color",[0.4 0.4 0.4], ...
    "LineStyle",":");
hold(ax1,"off");
grid(ax1,"on");
xlabel(ax1,"Actual VB");
ylabel(ax1,"Predicted VB");
legend(ax1,modelNames,"Location","northwest","Interpreter","none");
title(ax1,"High-wear prediction and alert boundary");

ax2 = nexttile;
metricToPlot = nan(numel(alertThresholds),numel(modelNames));
for thresholdIndex = 1:numel(alertThresholds)
    for modelIndex = 1:numel(modelNames)
        rowMask = classificationResults.AlertThreshold == ...
            alertThresholds(thresholdIndex) & ...
            classificationResults.Model == modelNames(modelIndex);
        metricToPlot(thresholdIndex,modelIndex) = ...
            classificationResults.Sensitivity(rowMask);
    end
end
bar(ax2,metricToPlot,"grouped");
grid(ax2,"on");
ylim(ax2,[0 1]);
xticks(ax2,1:numel(alertThresholds));
xticklabels(ax2,"VB ≥ " + string(alertThresholds));
ylabel(ax2,"Sensitivity (critical-wear recall)");
legend(ax2,modelNames,"Location","southeast","Interpreter","none");
title(ax2,"Critical-wear detection");

for axisHandle = [ax1 ax2]
    if ~isempty(axisHandle.Toolbar)
        axisHandle.Toolbar.Visible = "off";
    end
    disableDefaultInteractivity(axisHandle);
end
drawnow;

%% 7. Save
predictionTable = table(T.OriginalIndex,caseID,T.RunID,y, ...
    predProposed,predSafetyWeighted,'VariableNames', ...
    {'OriginalIndex','CaseID','RunID','ActualVB', ...
    'PrimaryPrediction','SafetyWeightedPrediction'});

writetable(overallResults,fullfile(outputFolder, ...
    "high_wear_overall.csv"));
writetable(wearBandResults,fullfile(outputFolder, ...
    "high_wear_bands.csv"));
writetable(classificationResults,fullfile(outputFolder, ...
    "critical_wear_classification.csv"));
writetable(caseResults,fullfile(outputFolder, ...
    "high_wear_case_metrics.csv"));
writetable(statisticalResults,fullfile(outputFolder, ...
    "high_wear_statistics.csv"));
writetable(hyperparameterResults,fullfile(outputFolder, ...
    "safety_weighted_hyperparameters.csv"));
writetable(predictionTable,fullfile(outputFolder, ...
    "high_wear_predictions.csv"));

save(fullfile(outputFolder,"high_wear_safety_validation.mat"), ...
    "overallResults","wearBandResults","classificationResults", ...
    "caseResults","statisticalResults","hyperparameterResults", ...
    "predictionTable","weightFactorGrid","lambdaGrid", ...
    "gateThresholdGrid","highWearTrainingThreshold");

exportgraphics(fig,fullfile(outputFolder, ...
    "high_wear_safety_validation.png"),"Resolution",300);
exportgraphics(fig,fullfile(outputFolder, ...
    "high_wear_safety_validation.pdf"),"ContentType","vector");

fprintf("\n========== High-wear overall results ==========\n");
disp(overallResults);
fprintf("\n========== Critical-wear classification ==========\n");
disp(classificationResults);
fprintf("\nFiles saved to:\n%s\n",outputFolder);


function [bestLambda,bestInnerPrediction] = selectWeightedBranch( ...
    X,y,caseID,outerTrainMask,lambdaGrid,observationWeight,wearThreshold)

trainingCases = unique(caseID(outerTrainMask));
validationLoss = nan(numel(lambdaGrid),1);
allInnerPrediction = nan(numel(y),numel(lambdaGrid));

for lambdaIndex = 1:numel(lambdaGrid)
    innerPrediction = nan(size(y));
    for innerIndex = 1:numel(trainingCases)
        validationMask = outerTrainMask & ...
            caseID == trainingCases(innerIndex);
        trainingMask = outerTrainMask & ...
            caseID ~= trainingCases(innerIndex);
        innerPrediction(validationMask) = fitWeightedRidge( ...
            X(trainingMask,:),y(trainingMask), ...
            observationWeight(trainingMask),X(validationMask,:), ...
            lambdaGrid(lambdaIndex));
    end
    allInnerPrediction(:,lambdaIndex) = innerPrediction;
    validationLoss(lambdaIndex) = balancedWearLoss( ...
        y(outerTrainMask),innerPrediction(outerTrainMask),wearThreshold);
end

[~,bestIndex] = min(validationLoss);
bestLambda = lambdaGrid(bestIndex);
bestInnerPrediction = allInnerPrediction(:,bestIndex);
end


function prediction = fitWeightedRidge( ...
    XTrain,yTrain,observationWeight,XTest,lambda)

featureMean = mean(XTrain,1);
featureStd = std(XTrain,0,1);
keepFeature = featureStd > 1e-12;
XTrain = XTrain(:,keepFeature);
XTest = XTest(:,keepFeature);
featureMean = featureMean(keepFeature);
featureStd = featureStd(keepFeature);
XTrain = (XTrain-featureMean)./featureStd;
XTest = (XTest-featureMean)./featureStd;

observationWeight = observationWeight(:);
responseMean = sum(observationWeight.*yTrain) / sum(observationWeight);
centeredResponse = yTrain-responseMean;
squareRootWeight = sqrt(observationWeight);
weightedX = XTrain.*squareRootWeight;
weightedY = centeredResponse.*squareRootWeight;
coefficient = (weightedX'*weightedX + ...
    lambda*eye(size(weightedX,2))) \ (weightedX'*weightedY);
prediction = responseMean + XTest*coefficient;
end


function loss = balancedWearLoss(y,prediction,wearThreshold)
lowMask = y < wearThreshold;
highMask = y >= wearThreshold;
if any(lowMask) && any(highMask)
    loss = 0.5*mean(abs(prediction(lowMask)-y(lowMask))) + ...
        0.5*mean(abs(prediction(highMask)-y(highMask)));
else
    loss = mean(abs(prediction-y));
end
end


function resultTable = regressionResultTable(y,predictionMatrix,modelNames)
numberOfModels = numel(modelNames);
MAE = nan(numberOfModels,1);
RMSE = nan(numberOfModels,1);
R2 = nan(numberOfModels,1);
MeanResidual = nan(numberOfModels,1);
P95AbsoluteError = nan(numberOfModels,1);
for modelIndex = 1:numberOfModels
    residual = predictionMatrix(:,modelIndex)-y;
    MAE(modelIndex) = mean(abs(residual));
    RMSE(modelIndex) = sqrt(mean(residual.^2));
    R2(modelIndex) = 1-sum(residual.^2)/sum((y-mean(y)).^2);
    MeanResidual(modelIndex) = mean(residual);
    P95AbsoluteError(modelIndex) = prctile(abs(residual),95);
end
resultTable = table(modelNames,MAE,RMSE,R2,MeanResidual, ...
    P95AbsoluteError);
end


function value = safeDivide(numerator,denominator)
if denominator == 0
    value = NaN;
else
    value = numerator/denominator;
end
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
