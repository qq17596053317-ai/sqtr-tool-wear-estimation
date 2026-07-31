clear;
clc;
close all;

%% Strict same-protocol PHM2010 comparison: Relief + RBF-SVR
rootFolder = fileparts(mfilename("fullpath"));
inputFile = fullfile(rootFolder,"phm2010_bundle_high_resolution.csv");
externalPredictionFile = fullfile(rootFolder,"external_validation_results", ...
    "external_validation_predictions.csv");
outputFolder = fullfile(rootFolder,"targeted_reviewer_experiments_20260719", ...
    "phm_relief_svr");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

rng(20260719,"twister");
fprintf("Reading PHM2010 source bundle...\n");
raw = readtable(inputFile,"VariableNamingRule","preserve");

sensorNames = ["force_x","force_y","force_z", ...
    "vibration_x","vibration_y","vibration_z"];
statNames = ["Mean","Maximum","Minimum","PeakToPeak", ...
    "Kurtosis","Variance","RMS"];
featureNames = strings(1,numel(sensorNames)*numel(statNames));
for sensorIndex = 1:numel(sensorNames)
    block = (sensorIndex-1)*numel(statNames)+(1:numel(statNames));
    featureNames(block) = sensorNames(sensorIndex)+"_"+statNames;
end

%% Run segmentation without using wear labels
experimentTag = string(raw.experiment_tag);
timestamp = double(raw.timestamp);
wear = double(raw.tool_wear);
groupID = zeros(height(raw),1);
nextGroup = 0;
experimentList = unique(experimentTag,"stable");
for experimentIndex = 1:numel(experimentList)
    rows = find(experimentTag == experimentList(experimentIndex));
    [sortedTime,order] = sort(timestamp(rows));
    rows = rows(order);
    positiveIncrement = diff(sortedTime);
    typicalCadence = median(positiveIncrement(positiveIncrement > 0),"omitnan");
    assert(isfinite(typicalCadence) && typicalCadence > 0, ...
        "Timestamp cadence could not be estimated.");
    boundaryThreshold = 0.1*typicalCadence;
    localRun = cumsum([true;positiveIncrement < boundaryThreshold]);
    groupID(rows) = nextGroup+localRun;
    nextGroup = nextGroup+max(localRun);
end
assert(all(groupID > 0),"Some source rows were not assigned to a cut.");

numberOfRuns = max(groupID);
X = nan(numberOfRuns,numel(featureNames));
y = nan(numberOfRuns,1);
groups = strings(numberOfRuns,1);
elapsedTime = nan(numberOfRuns,1);
sourceRowN = nan(numberOfRuns,1);

experimentStart = containers.Map("KeyType","char","ValueType","double");
for experimentIndex = 1:numel(experimentList)
    current = experimentList(experimentIndex);
    experimentStart(char(current)) = min(timestamp(experimentTag == current));
end

for runIndex = 1:numberOfRuns
    rows = find(groupID == runIndex);
    groups(runIndex) = experimentTag(rows(1));
    y(runIndex) = median(wear(rows),"omitnan");
    elapsedTime(runIndex) = median(timestamp(rows),"omitnan") - ...
        experimentStart(char(groups(runIndex)));
    sourceRowN(runIndex) = numel(rows);
    for sensorIndex = 1:numel(sensorNames)
        values = double(raw.(char(sensorNames(sensorIndex)))(rows));
        block = (sensorIndex-1)*numel(statNames)+(1:numel(statNames));
        X(runIndex,block) = gougamFeatures(values);
    end
end

sortTable = table(groups,elapsedTime,(1:numberOfRuns)', ...
    'VariableNames',cellstr(["Experiment","ElapsedTime","SourceOrder"]));
[~,order] = sortrows(sortTable,["Experiment","ElapsedTime","SourceOrder"]);
X = X(order,:);
y = y(order);
groups = groups(order);
elapsedTime = elapsedTime(order);
sourceRowN = sourceRowN(order);
runNumber = zeros(numberOfRuns,1);
experimentList = unique(groups,"stable");
for experimentIndex = 1:numel(experimentList)
    mask = groups == experimentList(experimentIndex);
    runNumber(mask) = (1:sum(mask))';
end
assert(numberOfRuns == 945,"Expected 945 PHM2010 cuts, obtained %d.",numberOfRuns);
assert(all(isfinite(X),"all") && all(isfinite(y)),"Non-finite PHM values detected.");

%% Nested leave-one-cutter-out validation
topKGrid = [5 10 15 20 30 42];
boxGrid = [0.1 1 10 100];
epsilonGrid = [0.001 0.005 0.01 0.02];
kernelScaleGrid = [0.1 1 10];
lambdaGrid = logspace(-4,4,17);

predictionMean = nan(numberOfRuns,1);
predictionRidge42 = nan(numberOfRuns,1);
predictionSVR42 = nan(numberOfRuns,1);
predictionReliefSVR = nan(numberOfRuns,1);

hyperRows = cell(numel(experimentList),10);
selectedRows = cell(0,5);

for outerIndex = 1:numel(experimentList)
    testGroup = experimentList(outerIndex);
    testMask = groups == testGroup;
    trainMask = ~testMask;
    fprintf("Outer test cutter %s: train=%d, test=%d\n", ...
        testGroup,sum(trainMask),sum(testMask));

    predictionMean(testMask) = mean(y(trainMask));

    bestLambda = tuneGroupedRidge(X(trainMask,:),y(trainMask), ...
        groups(trainMask),lambdaGrid);
    ridgeModel = fitRidgeLocal(X(trainMask,:),y(trainMask),bestLambda);
    predictionRidge42(testMask) = predictRidgeLocal(ridgeModel,X(testMask,:));

    [svrBox,svrEpsilon,svrScale,svrLoss] = tuneGroupedSVR( ...
        X(trainMask,:),y(trainMask),groups(trainMask), ...
        boxGrid,epsilonGrid,kernelScaleGrid);
    svrModel = fitSVRLocal(X(trainMask,:),y(trainMask), ...
        svrBox,svrEpsilon,svrScale,1:size(X,2));
    predictionSVR42(testMask) = predictSVRLocal(svrModel,X(testMask,:));

    [bestK,reliefBox,reliefEpsilon,reliefScale,reliefLoss] = ...
        tuneGroupedReliefSVR(X(trainMask,:),y(trainMask),groups(trainMask), ...
        topKGrid,boxGrid,epsilonGrid,kernelScaleGrid);
    [outerMu,outerSigma,XTrainZ,XTestZ] = standardizeTrainTest( ...
        X(trainMask,:),X(testMask,:));
    reliefNeighborN = min(10,sum(trainMask)-1);
    [outerRank,outerWeight] = relieff(XTrainZ,y(trainMask), ...
        reliefNeighborN,"method","regression");
    selectedIndex = outerRank(1:bestK);
    finalSVR = fitrsvm(XTrainZ(:,selectedIndex),y(trainMask), ...
        "KernelFunction","gaussian","BoxConstraint",reliefBox, ...
        "Epsilon",reliefEpsilon,"KernelScale",reliefScale, ...
        "Standardize",false);
    predictionReliefSVR(testMask) = predict(finalSVR,XTestZ(:,selectedIndex));

    hyperRows(outerIndex,:) = {testGroup,bestLambda,svrBox,svrEpsilon, ...
        svrScale,svrLoss,bestK,reliefBox,reliefEpsilon,reliefScale};
    for rankPosition = 1:bestK
        selectedRows(end+1,:) = {testGroup,rankPosition,selectedIndex(rankPosition), ...
            featureNames(selectedIndex(rankPosition)), ...
            outerWeight(selectedIndex(rankPosition))}; %#ok<SAGROW>
    end

    save(fullfile(outputFolder,"temporary_progress.mat"), ...
        "predictionMean","predictionRidge42","predictionSVR42", ...
        "predictionReliefSVR","hyperRows","selectedRows");
end

%% Add the existing strictly nested sensor+time ridge external result
existing = readtable(externalPredictionFile,"VariableNamingRule","preserve");
existing = existing(string(existing.Dataset) == "PHM2010",:);
predictionSensorTimeRidge = nan(numberOfRuns,1);
for rowIndex = 1:numberOfRuns
    match = string(existing.Experiment) == groups(rowIndex) & ...
        existing.RunIndex == runNumber(rowIndex);
    assert(sum(match) == 1,"Could not uniquely align PHM run %s-%d.", ...
        groups(rowIndex),runNumber(rowIndex));
    predictionSensorTimeRidge(rowIndex) = existing.SensorTimeRidge(match);
end

modelNames = ["TrainingMean";"Ridge42";"SVR42"; ...
    "ReliefSVR";"SensorTimeRidge"];
predictionMatrix = [predictionMean,predictionRidge42,predictionSVR42, ...
    predictionReliefSVR,predictionSensorTimeRidge];
summaryTable = regressionTable(y,predictionMatrix,modelNames,groups);
caseTable = caseMetricTable(y,predictionMatrix,modelNames,groups);

hyperparameterTable = cell2table(hyperRows,'VariableNames',cellstr( ...
    ["OuterTestCutter","RidgeLambda","SVRBox","SVREpsilon", ...
    "SVRKernelScale","SVRInnerMAE","ReliefTopK","ReliefSVRBox", ...
    "ReliefSVREpsilon","ReliefSVRKernelScale"]));
selectedFeatureTable = cell2table(selectedRows,'VariableNames',cellstr( ...
    ["OuterTestCutter","Rank","FeatureIndex","Feature","ReliefWeight"]));

predictionTable = table(groups,runNumber,elapsedTime,sourceRowN,y, ...
    predictionMean,predictionRidge42,predictionSVR42, ...
    predictionReliefSVR,predictionSensorTimeRidge, ...
    'VariableNames',cellstr(["Experiment","RunIndex","ElapsedTime", ...
    "TechnicalRowN","ActualWear","TrainingMean","Ridge42", ...
    "SVR42","ReliefSVR","SensorTimeRidge"]));

reliefCase = caseTable.MAE(caseTable.Model == "ReliefSVR");
proposedCase = caseTable.MAE(caseTable.Model == "SensorTimeRidge");
pairedDelta = proposedCase-reliefCase;
[pairedP,pairedStatistic] = safeSignrank(pairedDelta);
[ciLow,ciHigh,probabilityImprovement] = bootstrapCaseDifference( ...
    pairedDelta,10000,20260719);
statisticsTable = table(numel(pairedDelta),mean(reliefCase), ...
    mean(proposedCase),mean(pairedDelta),ciLow,ciHigh,pairedP, ...
    pairedStatistic,probabilityImprovement, ...
    'VariableNames',cellstr(["IndependentCutterN","ReliefSVRMacroMAE", ...
    "SensorTimeRidgeMacroMAE","MeanPairedDifference", ...
    "BootstrapCILower","BootstrapCIUpper","WilcoxonP", ...
    "SignedRankStatistic","BootstrapProbabilitySensorTimeImproves"]));

%% Publication-oriented figures
fig = figure("Visible","off","Color","w","Position",[100 100 1550 900]);
tl = tiledlayout(fig,2,2,"TileSpacing","compact","Padding","compact");
title(tl,"PHM2010 Strict Leave-One-Cutter-Out Comparison", ...
    "FontWeight","bold","FontSize",15);

ax1 = nexttile(tl);
bar(ax1,[summaryTable.MAE summaryTable.RMSE],"grouped");
grid(ax1,"on"); box(ax1,"on");
xticks(ax1,1:height(summaryTable));
xticklabels(ax1,replace(summaryTable.Model,"_"," "));
xtickangle(ax1,20);
ylabel(ax1,"Error (mm)");
legend(ax1,{"MAE","RMSE"},"Location","northwest");
title(ax1,"Overall predictive error");

ax2 = nexttile(tl);
caseNames = unique(groups,"stable");
reliefByCase = caseTable.MAE(caseTable.Model == "ReliefSVR");
timeByCase = caseTable.MAE(caseTable.Model == "SensorTimeRidge");
hold(ax2,"on");
for caseIndex = 1:numel(caseNames)
    plot(ax2,[1 2],[reliefByCase(caseIndex),timeByCase(caseIndex)], ...
        "-o","LineWidth",1.4,"MarkerFaceColor","auto");
end
hold(ax2,"off"); grid(ax2,"on"); box(ax2,"on");
xlim(ax2,[0.7 2.3]); xticks(ax2,[1 2]);
xticklabels(ax2,{"Relief + SVR","Sensor + time ridge"});
ylabel(ax2,"Cutter-level MAE (mm)");
title(ax2,sprintf("Paired cutters (n = %d)",numel(caseNames)));

ax3 = nexttile(tl);
[~,~,groupCode] = unique(groups,"stable");
scatter(ax3,y,predictionReliefSVR,18,groupCode,"filled", ...
    "MarkerFaceAlpha",0.55);
hold(ax3,"on");
bounds = [min([y;predictionReliefSVR]) max([y;predictionReliefSVR])];
plot(ax3,bounds,bounds,"k--","LineWidth",1.1);
hold(ax3,"off"); axis(ax3,"square"); grid(ax3,"on"); box(ax3,"on");
xlabel(ax3,"Actual wear (mm)"); ylabel(ax3,"Predicted wear (mm)");
title(ax3,sprintf("Relief + SVR: MAE %.4f, R^2 %.3f", ...
    summaryTable.MAE(summaryTable.Model == "ReliefSVR"), ...
    summaryTable.R2(summaryTable.Model == "ReliefSVR")));

ax4 = nexttile(tl);
scatter(ax4,y,predictionSensorTimeRidge,18,groupCode, ...
    "filled","MarkerFaceAlpha",0.55);
hold(ax4,"on");
bounds = [min([y;predictionSensorTimeRidge]) ...
    max([y;predictionSensorTimeRidge])];
plot(ax4,bounds,bounds,"k--","LineWidth",1.1);
hold(ax4,"off"); axis(ax4,"square"); grid(ax4,"on"); box(ax4,"on");
xlabel(ax4,"Actual wear (mm)"); ylabel(ax4,"Predicted wear (mm)");
title(ax4,sprintf("Sensor + time ridge: MAE %.4f, R^2 %.3f", ...
    summaryTable.MAE(summaryTable.Model == "SensorTimeRidge"), ...
    summaryTable.R2(summaryTable.Model == "SensorTimeRidge")));

exportgraphics(fig,fullfile(outputFolder,"phm_relief_svr_same_protocol.png"), ...
    "Resolution",300);
exportgraphics(fig,fullfile(outputFolder,"phm_relief_svr_same_protocol.pdf"), ...
    "ContentType","vector");
close(fig);

%% Save source data and methods record
writetable(summaryTable,fullfile(outputFolder,"phm_model_summary.csv"));
writetable(caseTable,fullfile(outputFolder,"phm_cutter_metrics.csv"));
writetable(predictionTable,fullfile(outputFolder,"phm_predictions.csv"));
writetable(hyperparameterTable,fullfile(outputFolder,"phm_nested_hyperparameters.csv"));
writetable(selectedFeatureTable,fullfile(outputFolder,"phm_relief_selected_features.csv"));
writetable(statisticsTable,fullfile(outputFolder,"phm_paired_statistics.csv"));
save(fullfile(outputFolder,"phm_relief_svr_same_protocol.mat"), ...
    "summaryTable","caseTable","predictionTable","hyperparameterTable", ...
    "selectedFeatureTable","statisticsTable","featureNames", ...
    "topKGrid","boxGrid","epsilonGrid","kernelScaleGrid","lambdaGrid");

writeReadme(outputFolder,summaryTable,statisticsTable,featureNames, ...
    topKGrid,boxGrid,epsilonGrid,kernelScaleGrid,lambdaGrid);
if isfile(fullfile(outputFolder,"temporary_progress.mat"))
    delete(fullfile(outputFolder,"temporary_progress.mat"));
end

fprintf("\n========== PHM2010 same-protocol summary ==========\n");
disp(summaryTable);
fprintf("\n========== Cutter-level paired inference ==========\n");
disp(statisticsTable);
fprintf("Results saved to:\n%s\n",outputFolder);


function f = gougamFeatures(x)
x = double(x(:));
x = x(isfinite(x));
mu = mean(x);
maximum = max(x);
minimum = min(x);
peakToPeak = maximum-minimum;
if numel(x) > 1
    varianceValue = var(x,0);
else
    varianceValue = 0;
end
rootMeanSquare = sqrt(mean(x.^2));
sigma = sqrt(varianceValue);
if sigma > 1e-12
    z = (x-mu)/sigma;
    kurtosisValue = mean(z.^4);
else
    kurtosisValue = 0;
end
f = [mu maximum minimum peakToPeak kurtosisValue varianceValue rootMeanSquare];
f(~isfinite(f)) = 0;
end


function [mu,sigma,XTrainZ,XTestZ] = standardizeTrainTest(XTrain,XTest)
mu = mean(XTrain,1,"omitnan");
sigma = std(XTrain,0,1,"omitnan");
mu(~isfinite(mu)) = 0;
sigma(~isfinite(sigma) | sigma < 1e-12) = 1;
for columnIndex = 1:size(XTrain,2)
    XTrain(~isfinite(XTrain(:,columnIndex)),columnIndex) = mu(columnIndex);
    XTest(~isfinite(XTest(:,columnIndex)),columnIndex) = mu(columnIndex);
end
XTrainZ = (XTrain-mu)./sigma;
XTestZ = (XTest-mu)./sigma;
end


function bestLambda = tuneGroupedRidge(X,y,groups,lambdaGrid)
groupList = unique(groups,"stable");
score = nan(numel(lambdaGrid),1);
for lambdaIndex = 1:numel(lambdaGrid)
    foldMAE = nan(numel(groupList),1);
    for foldIndex = 1:numel(groupList)
        validationMask = groups == groupList(foldIndex);
        trainingMask = ~validationMask;
        model = fitRidgeLocal(X(trainingMask,:),y(trainingMask), ...
            lambdaGrid(lambdaIndex));
        prediction = predictRidgeLocal(model,X(validationMask,:));
        foldMAE(foldIndex) = mean(abs(prediction-y(validationMask)));
    end
    score(lambdaIndex) = mean(foldMAE);
end
[~,bestIndex] = min(score);
bestLambda = lambdaGrid(bestIndex);
end


function model = fitRidgeLocal(X,y,lambda)
[mu,sigma,Xz,~] = standardizeTrainTest(X,zeros(0,size(X,2)));
meanY = mean(y);
beta = (Xz'*Xz+lambda*eye(size(Xz,2)))\(Xz'*(y-meanY));
model.mu = mu;
model.sigma = sigma;
model.meanY = meanY;
model.beta = beta;
end


function prediction = predictRidgeLocal(model,X)
for columnIndex = 1:size(X,2)
    X(~isfinite(X(:,columnIndex)),columnIndex) = model.mu(columnIndex);
end
Xz = (X-model.mu)./model.sigma;
prediction = model.meanY+Xz*model.beta;
end


function [bestBox,bestEpsilon,bestScale,bestLoss] = tuneGroupedSVR( ...
    X,y,groups,boxGrid,epsilonGrid,kernelScaleGrid)
groupList = unique(groups,"stable");
bestLoss = inf;
bestBox = boxGrid(1);
bestEpsilon = epsilonGrid(1);
bestScale = kernelScaleGrid(1);
for boxIndex = 1:numel(boxGrid)
    for epsilonIndex = 1:numel(epsilonGrid)
        for scaleIndex = 1:numel(kernelScaleGrid)
            foldMAE = nan(numel(groupList),1);
            for foldIndex = 1:numel(groupList)
                validationMask = groups == groupList(foldIndex);
                trainingMask = ~validationMask;
                model = fitSVRLocal(X(trainingMask,:),y(trainingMask), ...
                    boxGrid(boxIndex),epsilonGrid(epsilonIndex), ...
                    kernelScaleGrid(scaleIndex),1:size(X,2));
                prediction = predictSVRLocal(model,X(validationMask,:));
                foldMAE(foldIndex) = mean(abs(prediction-y(validationMask)));
            end
            currentLoss = mean(foldMAE);
            if currentLoss < bestLoss
                bestLoss = currentLoss;
                bestBox = boxGrid(boxIndex);
                bestEpsilon = epsilonGrid(epsilonIndex);
                bestScale = kernelScaleGrid(scaleIndex);
            end
        end
    end
end
end


function model = fitSVRLocal(X,y,boxValue,epsilonValue,scaleValue,selectedIndex)
[mu,sigma,Xz,~] = standardizeTrainTest(X,zeros(0,size(X,2)));
svr = fitrsvm(Xz(:,selectedIndex),y,"KernelFunction","gaussian", ...
    "BoxConstraint",boxValue,"Epsilon",epsilonValue, ...
    "KernelScale",scaleValue,"Standardize",false);
model.mu = mu;
model.sigma = sigma;
model.selectedIndex = selectedIndex;
model.svr = svr;
end


function prediction = predictSVRLocal(model,X)
for columnIndex = 1:size(X,2)
    X(~isfinite(X(:,columnIndex)),columnIndex) = model.mu(columnIndex);
end
Xz = (X-model.mu)./model.sigma;
prediction = predict(model.svr,Xz(:,model.selectedIndex));
end


function [bestK,bestBox,bestEpsilon,bestScale,bestLoss] = ...
    tuneGroupedReliefSVR(X,y,groups,topKGrid,boxGrid,epsilonGrid,kernelScaleGrid)
groupList = unique(groups,"stable");
bestLoss = inf;
bestK = topKGrid(1);
bestBox = boxGrid(1);
bestEpsilon = epsilonGrid(1);
bestScale = kernelScaleGrid(1);
for topKIndex = 1:numel(topKGrid)
    for boxIndex = 1:numel(boxGrid)
        for epsilonIndex = 1:numel(epsilonGrid)
            for scaleIndex = 1:numel(kernelScaleGrid)
                foldMAE = nan(numel(groupList),1);
                for foldIndex = 1:numel(groupList)
                    validationMask = groups == groupList(foldIndex);
                    trainingMask = ~validationMask;
                    [~,~,XTrainZ,XValidationZ] = standardizeTrainTest( ...
                        X(trainingMask,:),X(validationMask,:));
                    neighborN = min(10,sum(trainingMask)-1);
                    rankIndex = relieff(XTrainZ,y(trainingMask),neighborN, ...
                        "method","regression");
                    selected = rankIndex(1:topKGrid(topKIndex));
                    svr = fitrsvm(XTrainZ(:,selected),y(trainingMask), ...
                        "KernelFunction","gaussian", ...
                        "BoxConstraint",boxGrid(boxIndex), ...
                        "Epsilon",epsilonGrid(epsilonIndex), ...
                        "KernelScale",kernelScaleGrid(scaleIndex), ...
                        "Standardize",false);
                    prediction = predict(svr,XValidationZ(:,selected));
                    foldMAE(foldIndex) = mean(abs(prediction-y(validationMask)));
                end
                currentLoss = mean(foldMAE);
                if currentLoss < bestLoss
                    bestLoss = currentLoss;
                    bestK = topKGrid(topKIndex);
                    bestBox = boxGrid(boxIndex);
                    bestEpsilon = epsilonGrid(epsilonIndex);
                    bestScale = kernelScaleGrid(scaleIndex);
                end
            end
        end
    end
end
end


function result = regressionTable(y,predictionMatrix,modelNames,groups)
numberOfModels = numel(modelNames);
MAE = nan(numberOfModels,1);
RMSE = nan(numberOfModels,1);
R2 = nan(numberOfModels,1);
MacroMAE = nan(numberOfModels,1);
P95AbsoluteError = nan(numberOfModels,1);
MaximumAbsoluteError = nan(numberOfModels,1);
groupList = unique(groups,"stable");
for modelIndex = 1:numberOfModels
    residual = predictionMatrix(:,modelIndex)-y;
    MAE(modelIndex) = mean(abs(residual));
    RMSE(modelIndex) = sqrt(mean(residual.^2));
    R2(modelIndex) = 1-sum(residual.^2)/sum((y-mean(y)).^2);
    P95AbsoluteError(modelIndex) = prctile(abs(residual),95);
    MaximumAbsoluteError(modelIndex) = max(abs(residual));
    groupMAE = nan(numel(groupList),1);
    for groupIndex = 1:numel(groupList)
        mask = groups == groupList(groupIndex);
        groupMAE(groupIndex) = mean(abs(residual(mask)));
    end
    MacroMAE(modelIndex) = mean(groupMAE);
end
result = table(modelNames,MAE,RMSE,R2,MacroMAE, ...
    P95AbsoluteError,MaximumAbsoluteError,'VariableNames',cellstr( ...
    ["Model","MAE","RMSE","R2","MacroMAE", ...
    "P95AbsoluteError","MaximumAbsoluteError"]));
end


function result = caseMetricTable(y,predictionMatrix,modelNames,groups)
groupList = unique(groups,"stable");
rows = cell(numel(groupList)*numel(modelNames),6);
rowIndex = 0;
for groupIndex = 1:numel(groupList)
    mask = groups == groupList(groupIndex);
    for modelIndex = 1:numel(modelNames)
        rowIndex = rowIndex+1;
        residual = predictionMatrix(mask,modelIndex)-y(mask);
        rows(rowIndex,:) = {groupList(groupIndex),sum(mask), ...
            modelNames(modelIndex),mean(abs(residual)), ...
            sqrt(mean(residual.^2)),mean(residual)};
    end
end
result = cell2table(rows,'VariableNames',cellstr( ...
    ["Experiment","RunN","Model","MAE","RMSE","MeanResidual"]));
result.Experiment = string(result.Experiment);
result.Model = string(result.Model);
end


function [p,statistic] = safeSignrank(delta)
delta = delta(isfinite(delta) & delta ~= 0);
if isempty(delta)
    p = 1;
    statistic = 0;
    return;
end
try
    [p,~,stats] = signrank(delta,0,"method","exact");
catch
    [p,~,stats] = signrank(delta,0);
end
if isfield(stats,"signedrank")
    statistic = stats.signedrank;
else
    statistic = NaN;
end
end


function [lower,upper,probability] = bootstrapCaseDifference(delta,nBootstrap,seed)
stream = RandStream("mt19937ar","Seed",seed);
numberOfCases = numel(delta);
bootstrapMean = nan(nBootstrap,1);
for bootstrapIndex = 1:nBootstrap
    index = randi(stream,numberOfCases,numberOfCases,1);
    bootstrapMean(bootstrapIndex) = mean(delta(index));
end
limits = prctile(bootstrapMean,[2.5 97.5]);
lower = limits(1);
upper = limits(2);
probability = mean(bootstrapMean < 0);
end


function writeReadme(outputFolder,summaryTable,statisticsTable,featureNames, ...
    topKGrid,boxGrid,epsilonGrid,kernelScaleGrid,lambdaGrid)
fileID = fopen(fullfile(outputFolder,"README_PHM_RELIEF_SVR.txt"), ...
    "w","n","UTF-8");
assert(fileID > 0,"Could not create PHM README.");
cleanup = onCleanup(@() fclose(fileID)); %#ok<NASGU>
fprintf(fileID,"PHM2010 RELIEF + SVR SAME-PROTOCOL COMPARISON\n");
fprintf(fileID,"Generated with MATLAB %s\n\n",version);
fprintf(fileID,"Independent units and validation\n");
fprintf(fileID,"- Independent units: cutter sequences C1, C4 and C6 (n=3).\n");
fprintf(fileID,"- Technical rows were aggregated to 315 cutting runs per cutter.\n");
fprintf(fileID,"- Outer evaluation: leave one complete cutter out.\n");
fprintf(fileID,"- Inner evaluation: leave one of the two training cutters out.\n");
fprintf(fileID,"- Standardization, Relief ranking, feature-number selection and all hyperparameter selection were recomputed using training cutters only.\n");
fprintf(fileID,"- Wear labels from the outer test cutter were never used in preprocessing, selection or training.\n\n");
fprintf(fileID,"Gougam-aligned feature family\n");
fprintf(fileID,"- Six channels: force x/y/z and vibration x/y/z; acoustic emission excluded.\n");
fprintf(fileID,"- Seven features per channel: mean, maximum, minimum, peak-to-peak, kurtosis, variance and RMS (42 total).\n");
fprintf(fileID,"- Feature names: %s.\n\n",strjoin(featureNames,", "));
fprintf(fileID,"Search grids\n");
fprintf(fileID,"- Relief top-K: %s.\n",mat2str(topKGrid));
fprintf(fileID,"- SVR box constraint: %s.\n",mat2str(boxGrid));
fprintf(fileID,"- SVR epsilon: %s.\n",mat2str(epsilonGrid));
fprintf(fileID,"- SVR kernel scale: %s.\n",mat2str(kernelScaleGrid));
fprintf(fileID,"- Ridge lambda: %s.\n\n",mat2str(lambdaGrid));
fprintf(fileID,"Interpretation boundary\n");
fprintf(fileID,"- This is a method-aligned Relief + RBF-SVR baseline, not a claim of bit-for-bit reproduction of Gougam et al.\n");
fprintf(fileID,"- With only three independent cutters, cutter-level inferential power is intrinsically low. Record-level rows were not treated as independent replicates.\n");
fprintf(fileID,"- Raw MAE values should be compared only under this shared PHM2010 preprocessing and split protocol.\n\n");
fprintf(fileID,"Overall results\n");
for rowIndex = 1:height(summaryTable)
    fprintf(fileID,"%s: MAE=%.6f, RMSE=%.6f, R2=%.6f, macro-MAE=%.6f.\n", ...
        summaryTable.Model(rowIndex),summaryTable.MAE(rowIndex), ...
        summaryTable.RMSE(rowIndex),summaryTable.R2(rowIndex), ...
        summaryTable.MacroMAE(rowIndex));
end
fprintf(fileID,"\nPaired cutter comparison (SensorTimeRidge - ReliefSVR)\n");
fprintf(fileID,"n=%d, mean difference=%.6f, bootstrap 95%% CI=[%.6f, %.6f], exact Wilcoxon p=%.6g.\n", ...
    statisticsTable.IndependentCutterN,statisticsTable.MeanPairedDifference, ...
    statisticsTable.BootstrapCILower,statisticsTable.BootstrapCIUpper, ...
    statisticsTable.WilcoxonP);
end
