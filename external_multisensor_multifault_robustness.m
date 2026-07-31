clear; clc; close all;
paperRoot = fileparts(mfilename('fullpath'));
rng(20260717, 'twister');

baseDir = paperRoot;
previousDir = fullfile(baseDir, 'external_validation_results');
outDir = fullfile(baseDir, 'multifault_robustness_results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

logFile = fullfile(outDir, 'multifault_matlab_log.txt');
if exist(logFile, 'file')
    delete(logFile);
end
diary(logFile);
cleanupDiary = onCleanup(@() diary('off'));

fprintf('MATLAB version: %s\n', version);
fprintf('Multi-sensor multi-fault robustness experiment started: %s\n', char(datetime('now')));

matFile = fullfile(previousDir, 'external_validation_complete_results.mat');
assert(isfile(matFile), 'Previous external-validation MAT file is missing.');
load(matFile, 'nuaa', 'phm', 'lambdaGrid');

faultTypes = ["Saturation", "Dropout", "Bias", "Noise"];
severityMatrix = [ ...
    0.10 0.30 0.50 0.70; ...  % Saturation fraction
    0.10 0.30 0.50 0.70; ...  % Dropout fraction
    0.50 1.00 2.00 3.00; ...  % Bias in local signal SD
    0.25 0.50 1.00 2.00];     % Noise SD / local signal SD
severityUnits = ["fraction", "fraction", "signal_sd", "signal_sd"];
nBootstrap = 10000;

nuaaFile = fullfile(baseDir, 'nuaa_orthogonal_bundle_high_resolution.csv');
phmFile = fullfile(baseDir, 'phm2010_bundle_high_resolution.csv');

fprintf('\nBuilding NUAA fault variants...\n');
nuaaVariants = buildFaultVariants(nuaaFile, nuaa, "NUAA", ...
    ["force_z", "vibration1", "vibration2"], ...
    ["force_z", "vibration_x", "vibration_y"], ...
    faultTypes, severityMatrix);

fprintf('\nBuilding PHM2010 fault variants...\n');
phmVariants = buildFaultVariants(phmFile, phm, "PHM2010", ...
    ["force_z", "vibration_x", "vibration_y"], ...
    ["force_z", "vibration_x", "vibration_y"], ...
    faultTypes, severityMatrix);

fprintf('\nRunning NUAA nested multi-fault validation...\n');
nuaaResult = validateMultiFault(nuaa, nuaaVariants, lambdaGrid, ...
    faultTypes, severityMatrix, severityUnits);

fprintf('\nRunning PHM2010 nested multi-fault validation...\n');
phmResult = validateMultiFault(phm, phmVariants, lambdaGrid, ...
    faultTypes, severityMatrix, severityUnits);

scenarioSummary = [nuaaResult.scenarioSummary; phmResult.scenarioSummary];
detectionSummary = [nuaaResult.detectionSummary; phmResult.detectionSummary];
cleanSummary = [nuaaResult.cleanSummary; phmResult.cleanSummary];
caseAggregate = [nuaaResult.caseAggregate; phmResult.caseAggregate];
predictionSourceData = [nuaaResult.predictionSourceData; phmResult.predictionSourceData];
statisticsTable = makeRobustnessStatistics(nuaaResult, phmResult, nBootstrap);

writetable(scenarioSummary, fullfile(outDir, 'multifault_scenario_summary.csv'));
writetable(detectionSummary, fullfile(outDir, 'multifault_detection_summary.csv'));
writetable(cleanSummary, fullfile(outDir, 'multifault_clean_gating_summary.csv'));
writetable(caseAggregate, fullfile(outDir, 'multifault_case_aggregate.csv'));
writetable(predictionSourceData, fullfile(outDir, 'multifault_prediction_source_data.csv'));
writetable(statisticsTable, fullfile(outDir, 'multifault_statistics.csv'));

fprintf('\n===== Clean-data gating cost =====\n');
disp(cleanSummary);
fprintf('\n===== Experiment-level aggregate robustness =====\n');
disp(caseAggregate);
fprintf('\n===== Robustness statistics =====\n');
disp(statisticsTable);

makeRobustnessGridFigure(scenarioSummary, cleanSummary, faultTypes, outDir);
makeDetectionFigure(detectionSummary, faultTypes, outDir);
makeCasePairedFigure(caseAggregate, statisticsTable, outDir);
makeHeadlineFigure(scenarioSummary, detectionSummary, caseAggregate, outDir);

save(fullfile(outDir, 'multifault_complete_results.mat'), ...
    'nuaaResult', 'phmResult', 'scenarioSummary', 'detectionSummary', ...
    'cleanSummary', 'caseAggregate', 'statisticsTable', ...
    'faultTypes', 'severityMatrix', 'severityUnits', 'lambdaGrid', '-v7.3');

writeMultifaultReadme(outDir, faultTypes, severityMatrix, severityUnits, ...
    cleanSummary, statisticsTable);

fprintf('\nAll multi-fault robustness results saved to:\n%s\n', outDir);
fprintf('Experiment finished: %s\n', char(datetime('now')));


function V = buildFaultVariants(filePath, D, datasetName, rawSensorNames, ...
        displaySensorNames, faultTypes, severityMatrix)
    T = readtable(filePath, 'VariableNamingRule', 'preserve');
    groupId = makeRawGroupId(T, datasetName);
    n = height(D.runTable);
    p = size(D.sensorX, 2);
    nSensors = numel(rawSensorNames);
    nFaults = numel(faultTypes);
    nLevels = size(severityMatrix, 2);
    nStats = 9;
    nQuality = 4;

    sensorIndices = zeros(1, nSensors);
    featureBlocks = cell(1, nSensors);
    for s = 1:nSensors
        sensorIndices(s) = find(D.sensorNames == rawSensorNames(s), 1);
        assert(sensorIndices(s) > 0, 'Monitored sensor not found: %s', rawSensorNames(s));
        featureBlocks{s} = (sensorIndices(s)-1)*nStats + (1:nStats);
    end

    Xvariants = repmat(D.sensorX, [1 1 nSensors nFaults nLevels]);
    cleanQuality = NaN(n, nSensors, nQuality);
    targetQuality = NaN(n, nSensors, nFaults, nLevels, nQuality);

    for i = 1:n
        if mod(i, max(1, floor(n/10))) == 0
            fprintf('  %s corruption construction: %d/%d runs\n', datasetName, i, n);
        end
        rows = find(groupId == D.runTable.SourceGroup(i));
        assert(~isempty(rows), 'No raw rows found for run %d.', i);
        for s = 1:nSensors
            x = double(T.(char(rawSensorNames(s)))(rows));
            x = x(isfinite(x));
            qClean = rawQualityDescriptors(x);
            cleanQuality(i, s, :) = reshape(qClean, 1, 1, []);
            block = featureBlocks{s};
            for f = 1:nFaults
                for l = 1:nLevels
                    xc = injectFault(x, faultTypes(f), severityMatrix(f,l));
                    Xvariants(i, block, s, f, l) = signalFeatures(xc);
                    qTarget = rawQualityDescriptors(xc);
                    targetQuality(i, s, f, l, :) = reshape(qTarget, 1, 1, 1, 1, []);
                end
            end
        end
    end

    V.name = datasetName;
    V.rawSensorNames = rawSensorNames;
    V.displaySensorNames = displaySensorNames;
    V.sensorIndices = sensorIndices;
    V.featureBlocks = featureBlocks;
    V.Xvariants = Xvariants;
    V.cleanQuality = cleanQuality;
    V.targetQuality = targetQuality;
end


function groupId = makeRawGroupId(T, datasetName)
    tag = string(T.experiment_tag);
    timestamp = double(T.timestamp);
    if datasetName == "NUAA"
        [groupId, ~, ~] = findgroups(tag, double(T.experiment_csv_n));
    else
        % Reconstruct PHM2010 cuts from timestamp cadence only.  This must
        % match external_validation_nuaa_phm2010.m and deliberately avoids
        % using the target tool-wear label to define run boundaries.
        groupId = zeros(height(T), 1);
        nextGroup = 0;
        tags = unique(tag, 'stable');
        for i = 1:numel(tags)
            rows = find(tag == tags(i));
            [sortedTimestamp, order] = sort(timestamp(rows));
            rows = rows(order);
            positiveIncrement = diff(sortedTimestamp);
            typicalCadence = median( ...
                positiveIncrement(positiveIncrement > 0), 'omitnan');
            assert(isfinite(typicalCadence) && typicalCadence > 0, ...
                'Could not estimate timestamp cadence for %s.', tags(i));
            boundaryThreshold = 0.1 * typicalCadence;
            localRun = cumsum([true; positiveIncrement < boundaryThreshold]);
            groupId(rows) = nextGroup + localRun;
            nextGroup = nextGroup + max(localRun);
        end
    end
end


function xc = injectFault(x, faultType, severity)
    xc = double(x(:));
    n = numel(xc);
    if n == 0
        return;
    end
    localScale = std(xc, 0, 'omitnan');
    if ~isfinite(localScale) || localScale < 1e-10
        localScale = max(1e-6, 0.01 * max(abs(xc), [], 'omitnan'));
    end
    switch faultType
        case "Saturation"
            k = min(n, max(1, round(severity * n)));
            idx = unique(round(linspace(1, n, k)));
            xc(idx) = max(xc, [], 'omitnan');
        case "Dropout"
            k = min(n, max(1, round(severity * n)));
            idx = unique(round(linspace(1, n, k)));
            xc(idx) = 0;
        case "Bias"
            xc = xc + severity * localScale;
        case "Noise"
            xc = xc + severity * localScale .* randn(size(xc));
        otherwise
            error('Unknown fault type: %s', faultType);
    end
end


function q = rawQualityDescriptors(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        q = [1 1 1 0];
        return;
    end
    tolerance = max(1e-12, 1e-8 * max(1, max(abs(x))));
    lowerRailRatio = mean(abs(x - min(x)) <= tolerance);
    upperRailRatio = mean(abs(x - max(x)) <= tolerance);
    railRatio = max(lowerRailRatio, upperRailRatio);
    zeroRatio = mean(abs(x) <= tolerance);
    if numel(x) > 1
        dx = diff(x);
        flatRatio = mean(abs(dx) <= tolerance);
        diffRms = sqrt(mean(dx.^2));
    else
        flatRatio = 1;
        diffRms = 0;
    end
    signalStd = std(x, 0);
    diffToStd = diffRms / max(signalStd, 1e-10);
    q = [railRatio, zeroRatio, flatRatio, diffToStd];
    q(~isfinite(q)) = 0;
end


function f = signalFeatures(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        f = zeros(1, 9);
        return;
    end
    mu = mean(x);
    rootMeanSquare = sqrt(mean(x.^2));
    if numel(x) > 1
        sigma = std(x, 0);
    else
        sigma = 0;
    end
    peakToPeak = max(x) - min(x);
    med = median(x);
    madValue = median(abs(x - med));
    if sigma > max(eps(abs(mu)), 1e-12)
        z = (x - mu) ./ sigma;
        skewValue = mean(z.^3);
        kurtValue = mean(z.^4);
    else
        skewValue = 0;
        kurtValue = 0;
    end
    crestFactor = max(abs(x)) / max(rootMeanSquare, eps);
    f = [mu, rootMeanSquare, sigma, peakToPeak, skewValue, ...
        kurtValue, med, madValue, crestFactor];
    f(~isfinite(f)) = 0;
end


function R = validateMultiFault(D, V, lambdaGrid, faultTypes, severityMatrix, severityUnits)
    y = D.y;
    groups = D.experiment;
    cases = unique(groups, 'stable');
    n = numel(y);
    nSensors = numel(V.displaySensorNames);
    nFaults = numel(faultTypes);
    nLevels = size(severityMatrix, 2);

    XfullClean = [D.processX, D.sensorX, D.timeX];
    keepMasks = cell(1, nSensors);
    XdropClean = cell(1, nSensors);
    for s = 1:nSensors
        keepMasks{s} = true(1, size(D.sensorX, 2));
        keepMasks{s}(V.featureBlocks{s}) = false;
        XdropClean{s} = [D.processX, D.sensorX(:, keepMasks{s}), D.timeX];
    end

    predCleanFull = NaN(n,1);
    predCleanAuto = NaN(n,1);
    cleanFlag = zeros(n,1);
    predFull = NaN(n,nSensors,nFaults,nLevels);
    predAuto = NaN(n,nSensors,nFaults,nLevels);
    predOracle = NaN(n,nSensors,nFaults,nLevels);
    detectedSensor = zeros(n,nSensors,nFaults,nLevels);

    for c = 1:numel(cases)
        testCase = cases(c);
        isTest = groups == testCase;
        isTrain = ~isTest;
        fprintf('  Outer test %s: training %d runs, testing %d runs\n', ...
            testCase, sum(isTrain), sum(isTest));

        lambdaFull = groupedTuneLambda(XfullClean(isTrain,:), y(isTrain), ...
            groups(isTrain), lambdaGrid);
        fullModel = fitRidge(XfullClean(isTrain,:), y(isTrain), lambdaFull);

        dropModels = cell(1,nSensors);
        for s = 1:nSensors
            lambdaDrop = groupedTuneLambda(XdropClean{s}(isTrain,:), y(isTrain), ...
                groups(isTrain), lambdaGrid);
            dropModels{s} = fitRidge(XdropClean{s}(isTrain,:), y(isTrain), lambdaDrop);
        end

        detector = fitQualityDetector(D.sensorX(isTrain,:), ...
            V.cleanQuality(isTrain,:,:), V.featureBlocks);

        cleanFull = predictRidge(fullModel, XfullClean(isTest,:));
        cleanDrop = NaN(sum(isTest), nSensors);
        for s = 1:nSensors
            cleanDrop(:,s) = predictRidge(dropModels{s}, XdropClean{s}(isTest,:));
        end
        cleanFlags = detectQualityFaults(D.sensorX(isTest,:), ...
            V.cleanQuality(isTest,:,:), detector, V.featureBlocks);
        cleanAuto = routePredictions(cleanFull, cleanDrop, cleanFlags);
        predCleanFull(isTest) = cleanFull;
        predCleanAuto(isTest) = cleanAuto;
        cleanFlag(isTest) = cleanFlags;

        for targetSensor = 1:nSensors
            for f = 1:nFaults
                for l = 1:nLevels
                    Xvariant = squeeze(V.Xvariants(isTest,:,targetSensor,f,l));
                    if sum(isTest) == 1
                        Xvariant = reshape(Xvariant,1,[]);
                    end
                    Xfull = [D.processX(isTest,:), Xvariant, D.timeX(isTest,:)];
                    fullPrediction = predictRidge(fullModel, Xfull);
                    dropPrediction = NaN(sum(isTest), nSensors);
                    for routeSensor = 1:nSensors
                        Xdrop = [D.processX(isTest,:), ...
                            Xvariant(:,keepMasks{routeSensor}), D.timeX(isTest,:)];
                        dropPrediction(:,routeSensor) = predictRidge( ...
                            dropModels{routeSensor}, Xdrop);
                    end

                    scenarioQuality = V.cleanQuality(isTest,:,:);
                    targetQ = squeeze(V.targetQuality(isTest,targetSensor,f,l,:));
                    if sum(isTest) == 1
                        targetQ = reshape(targetQ,1,[]);
                    end
                    scenarioQuality(:,targetSensor,:) = reshape(targetQ, ...
                        sum(isTest),1,size(targetQ,2));
                    flags = detectQualityFaults(Xvariant, scenarioQuality, ...
                        detector, V.featureBlocks);
                    autoPrediction = routePredictions(fullPrediction, dropPrediction, flags);

                    predFull(isTest,targetSensor,f,l) = fullPrediction;
                    predAuto(isTest,targetSensor,f,l) = autoPrediction;
                    predOracle(isTest,targetSensor,f,l) = dropPrediction(:,targetSensor);
                    detectedSensor(isTest,targetSensor,f,l) = flags;
                end
            end
        end
    end

    cleanSummary = makeCleanSummary(D.name, y, groups, predCleanFull, ...
        predCleanAuto, cleanFlag);
    [scenarioSummary, detectionSummary] = makeScenarioTables(D, V, faultTypes, ...
        severityMatrix, severityUnits, predFull, predAuto, predOracle, detectedSensor, ...
        predCleanFull, cleanFlag);
    caseAggregate = makeCaseAggregate(D.name, y, groups, predCleanFull, ...
        predFull, predAuto, predOracle);
    predictionSourceData = makePredictionSourceData(D, V, faultTypes, ...
        severityMatrix, severityUnits, predFull, predAuto, predOracle, detectedSensor);

    R.name = D.name;
    R.groups = groups;
    R.y = y;
    R.cleanSummary = cleanSummary;
    R.scenarioSummary = scenarioSummary;
    R.detectionSummary = detectionSummary;
    R.caseAggregate = caseAggregate;
    R.predictionSourceData = predictionSourceData;
end


function detector = fitQualityDetector(Xsensor, cleanQuality, featureBlocks)
    nSensors = numel(featureBlocks);
    selectedStats = [1 2 3 4 7 8 9];
    detector = struct;
    detector.center = cell(1,nSensors);
    detector.scale = cell(1,nSensors);
    detector.threshold = zeros(1,nSensors);
    for s = 1:nSensors
        rawQ = squeeze(cleanQuality(:,s,:));
        Q = [Xsensor(:,featureBlocks{s}(selectedStats)), rawQ];
        center = median(Q,1,'omitnan');
        scale = 1.4826 * median(abs(Q-center),1,'omitnan');
        fallback = std(Q,0,1,'omitnan');
        bad = ~isfinite(scale) | scale < 1e-9;
        scale(bad) = fallback(bad);
        scale(~isfinite(scale) | scale < 1e-9) = 1e-9;
        score = max(abs((Q-center)./scale),[],2);
        threshold = max(4, prctile(score,99));
        detector.center{s} = center;
        detector.scale{s} = scale;
        detector.threshold(s) = threshold;
    end
end


function flags = detectQualityFaults(Xsensor, quality, detector, featureBlocks)
    n = size(Xsensor,1);
    nSensors = numel(featureBlocks);
    selectedStats = [1 2 3 4 7 8 9];
    normalizedScore = zeros(n,nSensors);
    for s = 1:nSensors
        rawQ = squeeze(quality(:,s,:));
        if n == 1
            rawQ = reshape(rawQ,1,[]);
        end
        Q = [Xsensor(:,featureBlocks{s}(selectedStats)), rawQ];
        score = max(abs((Q-detector.center{s})./detector.scale{s}),[],2);
        normalizedScore(:,s) = score ./ detector.threshold(s);
    end
    [bestScore, flags] = max(normalizedScore,[],2);
    flags(bestScore <= 1) = 0;
end


function routed = routePredictions(fullPrediction, dropPrediction, flags)
    routed = fullPrediction;
    for s = 1:size(dropPrediction,2)
        idx = flags == s;
        routed(idx) = dropPrediction(idx,s);
    end
end


function lambda = groupedTuneLambda(X, y, groups, lambdaGrid)
    cases = unique(groups,'stable');
    if numel(cases) < 2
        lambda = 1;
        return;
    end
    scores = NaN(numel(lambdaGrid),1);
    for k = 1:numel(lambdaGrid)
        foldMAE = NaN(numel(cases),1);
        for c = 1:numel(cases)
            isVal = groups == cases(c);
            isTrain = ~isVal;
            model = fitRidge(X(isTrain,:),y(isTrain),lambdaGrid(k));
            pred = predictRidge(model,X(isVal,:));
            foldMAE(c) = mean(abs(pred-y(isVal)));
        end
        scores(k) = mean(foldMAE,'omitnan');
    end
    [~,best] = min(scores);
    lambda = lambdaGrid(best);
end


function model = fitRidge(X,y,lambda)
    X = double(X); y = double(y(:));
    muX = mean(X,1,'omitnan'); muX(~isfinite(muX)) = 0;
    for j = 1:size(X,2)
        bad = ~isfinite(X(:,j)); X(bad,j) = muX(j);
    end
    sigmaX = std(X,0,1);
    sigmaX(~isfinite(sigmaX) | sigmaX < 1e-12) = 1;
    Xz = (X-muX)./sigmaX;
    muY = mean(y); yc = y-muY;
    beta = (Xz'*Xz + lambda*eye(size(Xz,2))) \ (Xz'*yc);
    model.muX = muX; model.sigmaX = sigmaX;
    model.muY = muY; model.beta = beta; model.lambda = lambda;
end


function yhat = predictRidge(model,X)
    X = double(X);
    for j = 1:size(X,2)
        bad = ~isfinite(X(:,j)); X(bad,j) = model.muX(j);
    end
    yhat = model.muY + ((X-model.muX)./model.sigmaX)*model.beta;
end


function T = makeCleanSummary(datasetName,y,groups,predFull,predAuto,cleanFlag)
    modelNames = ["FullClean";"AutoGatedClean"];
    predictions = [predFull,predAuto];
    dataset = repmat(datasetName,2,1);
    MAE = zeros(2,1); RMSE = zeros(2,1); R2 = zeros(2,1); MacroMAE = zeros(2,1);
    for m = 1:2
        met = regressionMetrics(y,predictions(:,m),groups);
        MAE(m)=met.MAE; RMSE(m)=met.RMSE; R2(m)=met.R2; MacroMAE(m)=met.MacroMAE;
    end
    CleanFalseAlarmRate = repmat(mean(cleanFlag>0),2,1);
    T = table(dataset,modelNames,MAE,RMSE,R2,MacroMAE,CleanFalseAlarmRate, ...
        'VariableNames',{'Dataset','Model','MAE','RMSE','R2','MacroMAE','CleanFalseAlarmRate'});
end


function [scenarioT,detectionT] = makeScenarioTables(D,V,faultTypes,severityMatrix, ...
        severityUnits,predFull,predAuto,predOracle,detectedSensor,predCleanFull,cleanFlag)
    nSensors = numel(V.displaySensorNames); nFaults = numel(faultTypes);
    nLevels = size(severityMatrix,2);
    modelNames = ["FullModel";"AutoGate";"OracleGate"];
    nScenarioRows = nSensors*nFaults*nLevels*numel(modelNames);
    scenarioRows = cell(nScenarioRows,14);
    detectionRows = cell(nSensors*nFaults*nLevels,12);
    cleanMetrics = regressionMetrics(D.y,predCleanFull,D.experiment);
    sr = 0; dr = 0;
    for s = 1:nSensors
        for f = 1:nFaults
            for l = 1:nLevels
                predictions = [predFull(:,s,f,l),predAuto(:,s,f,l),predOracle(:,s,f,l)];
                for m = 1:3
                    sr = sr+1;
                    met = regressionMetrics(D.y,predictions(:,m),D.experiment);
                    scenarioRows(sr,:) = {D.name,V.displaySensorNames(s),faultTypes(f),l, ...
                        severityMatrix(f,l),severityUnits(f),modelNames(m),met.MAE,met.RMSE, ...
                        met.R2,met.MacroMAE,met.MAE/cleanMetrics.MAE, ...
                        met.P95AbsoluteError,met.MaximumAbsoluteError};
                end
                dr = dr+1;
                flags = detectedSensor(:,s,f,l);
                detectionRows(dr,:) = {D.name,V.displaySensorNames(s),faultTypes(f),l, ...
                    severityMatrix(f,l),severityUnits(f),mean(flags>0),mean(flags==s), ...
                    mean(flags>0 & flags~=s),mean(cleanFlag>0),sum(flags==s),numel(flags)};
            end
        end
    end
    scenarioT = cell2table(scenarioRows,'VariableNames', ...
        {'Dataset','FaultSensor','FaultType','SeverityLevel','SeverityValue','SeverityUnit', ...
        'Model','MAE','RMSE','R2','MacroMAE','NormalizedMAE','P95AbsoluteError','MaximumAbsoluteError'});
    detectionT = cell2table(detectionRows,'VariableNames', ...
        {'Dataset','FaultSensor','FaultType','SeverityLevel','SeverityValue','SeverityUnit', ...
        'AnyAlarmRate','CorrectSensorDetectionRate','WrongSensorAlarmRate','CleanFalseAlarmRate', ...
        'CorrectDetections','RunLevelN'});
    stringVars = {'Dataset','FaultSensor','FaultType','SeverityUnit','Model'};
    for i = 1:numel(stringVars)
        scenarioT.(stringVars{i}) = string(scenarioT.(stringVars{i}));
    end
    stringVars = {'Dataset','FaultSensor','FaultType','SeverityUnit'};
    for i = 1:numel(stringVars)
        detectionT.(stringVars{i}) = string(detectionT.(stringVars{i}));
    end
end


function T = makeCaseAggregate(datasetName,y,groups,predCleanFull,predFull,predAuto,predOracle)
    cases = unique(groups,'stable'); nCases = numel(cases);
    dataset = repmat(datasetName,nCases,1);
    RunLevelN = zeros(nCases,1); CleanMAE = zeros(nCases,1);
    FullMeanCorruptMAE = zeros(nCases,1); AutoMeanCorruptMAE = zeros(nCases,1);
    OracleMeanCorruptMAE = zeros(nCases,1);
    FullRelativeMAE = zeros(nCases,1); AutoRelativeMAE = zeros(nCases,1);
    OracleRelativeMAE = zeros(nCases,1);
    for c = 1:nCases
        idx = groups==cases(c); RunLevelN(c)=sum(idx);
        CleanMAE(c)=mean(abs(predCleanFull(idx)-y(idx)));
        actual = reshape(y(idx),[],1,1,1);
        FullMeanCorruptMAE(c)=mean(abs(predFull(idx,:,:,:)-actual),'all');
        AutoMeanCorruptMAE(c)=mean(abs(predAuto(idx,:,:,:)-actual),'all');
        OracleMeanCorruptMAE(c)=mean(abs(predOracle(idx,:,:,:)-actual),'all');
        base = max(CleanMAE(c),1e-8);
        FullRelativeMAE(c)=FullMeanCorruptMAE(c)/base;
        AutoRelativeMAE(c)=AutoMeanCorruptMAE(c)/base;
        OracleRelativeMAE(c)=OracleMeanCorruptMAE(c)/base;
    end
    T = table(dataset,cases,RunLevelN,CleanMAE,FullMeanCorruptMAE,AutoMeanCorruptMAE, ...
        OracleMeanCorruptMAE,FullRelativeMAE,AutoRelativeMAE,OracleRelativeMAE, ...
        'VariableNames',{'Dataset','Experiment','RunLevelN','CleanMAE','FullMeanCorruptMAE', ...
        'AutoMeanCorruptMAE','OracleMeanCorruptMAE','FullRelativeMAE','AutoRelativeMAE','OracleRelativeMAE'});
end


function T = makePredictionSourceData(D,V,faultTypes,severityMatrix,severityUnits, ...
        predFull,predAuto,predOracle,detectedSensor)
    n = numel(D.y); nSensors = numel(V.displaySensorNames);
    nFaults = numel(faultTypes); nLevels = size(severityMatrix,2);
    total = n*nSensors*nFaults*nLevels;
    Dataset = strings(total,1); Experiment = strings(total,1); FaultSensor = strings(total,1);
    FaultType = strings(total,1); SeverityUnit = strings(total,1);
    RunIndex = zeros(total,1); ActualWear = zeros(total,1); SeverityLevel = zeros(total,1);
    SeverityValue = zeros(total,1); FullPrediction = zeros(total,1);
    AutoPrediction = zeros(total,1); OraclePrediction = zeros(total,1);
    DetectedSensorIndex = zeros(total,1); DetectedSensorName = strings(total,1);
    flagNames = ["None",V.displaySensorNames]; r=0;
    for s=1:nSensors
        for f=1:nFaults
            for l=1:nLevels
                for i=1:n
                    r=r+1; Dataset(r)=D.name; Experiment(r)=D.experiment(i);
                    RunIndex(r)=D.runTable.RunIndex(i); ActualWear(r)=D.y(i);
                    FaultSensor(r)=V.displaySensorNames(s); FaultType(r)=faultTypes(f);
                    SeverityLevel(r)=l; SeverityValue(r)=severityMatrix(f,l);
                    SeverityUnit(r)=severityUnits(f); FullPrediction(r)=predFull(i,s,f,l);
                    AutoPrediction(r)=predAuto(i,s,f,l); OraclePrediction(r)=predOracle(i,s,f,l);
                    DetectedSensorIndex(r)=detectedSensor(i,s,f,l);
                    DetectedSensorName(r)=flagNames(DetectedSensorIndex(r)+1);
                end
            end
        end
    end
    T = table(Dataset,Experiment,RunIndex,ActualWear,FaultSensor,FaultType,SeverityLevel, ...
        SeverityValue,SeverityUnit,FullPrediction,AutoPrediction,OraclePrediction, ...
        DetectedSensorIndex,DetectedSensorName);
end


function met = regressionMetrics(y,pred,groups)
    residual=pred-y; absError=abs(residual);
    met.MAE=mean(absError); met.RMSE=sqrt(mean(residual.^2));
    denominator=sum((y-mean(y)).^2);
    met.R2=1-sum(residual.^2)/denominator;
    cases=unique(groups,'stable'); caseMAE=zeros(numel(cases),1);
    for c=1:numel(cases), caseMAE(c)=mean(absError(groups==cases(c))); end
    met.MacroMAE=mean(caseMAE); met.P95AbsoluteError=prctile(absError,95);
    met.MaximumAbsoluteError=max(absError);
end


function T = makeRobustnessStatistics(nuaaResult,phmResult,nBootstrap)
    results={nuaaResult,phmResult}; rows=cell(3,14); rawP=NaN(3,1);
    allFull=[]; allAuto=[]; allOracle=[];
    for d=1:2
        C=results{d}.caseAggregate;
        delta=C.AutoRelativeMAE-C.FullRelativeMAE;
        [p,stat]=safeSignrank(delta); rawP(d)=p;
        [lo,hi,prob]=bootstrapDifference(delta,nBootstrap);
        rows(d,1:13)={results{d}.name,height(C),mean(C.FullRelativeMAE), ...
            mean(C.AutoRelativeMAE),mean(C.OracleRelativeMAE),mean(delta), ...
            100*(mean(C.FullRelativeMAE)-mean(C.AutoRelativeMAE))/mean(C.FullRelativeMAE), ...
            sum(delta<0),sum(delta>0),p,stat,lo,hi};
        rows{d,14}=prob;
        allFull=[allFull;C.FullRelativeMAE]; %#ok<AGROW>
        allAuto=[allAuto;C.AutoRelativeMAE]; %#ok<AGROW>
        allOracle=[allOracle;C.OracleRelativeMAE]; %#ok<AGROW>
    end
    delta=allAuto-allFull; [p,stat]=safeSignrank(delta);
    rawP(3)=p;
    [lo,hi,prob]=bootstrapDifference(delta,nBootstrap);
    rows(3,1:14)={"Combined",numel(delta),mean(allFull),mean(allAuto),mean(allOracle), ...
        mean(delta),100*(mean(allFull)-mean(allAuto))/mean(allFull),sum(delta<0), ...
        sum(delta>0),p,stat,lo,hi,prob};
    T=cell2table(rows,'VariableNames',{'Dataset','IndependentExperimentN', ...
        'FullRelativeMAE','AutoRelativeMAE','OracleRelativeMAE','MeanPairedDifference', ...
        'RelativeMAEImprovementPercent','ExperimentsImproved','ExperimentsWorsened', ...
        'WilcoxonP','SignedRankStatistic','BootstrapCILower','BootstrapCIUpper', ...
        'BootstrapProbabilityImprovement'});
    adjusted=holmAdjust(rawP);
    T.Dataset=string(T.Dataset); T.HolmAdjustedP=adjusted;
    T.SignificantAfterHolm=T.HolmAdjustedP<0.05;
end


function [p,statistic]=safeSignrank(delta)
    delta=delta(isfinite(delta)&delta~=0);
    if isempty(delta), p=1; statistic=0; return; end
    try
        [p,~,stats]=signrank(delta,0,'method','exact');
    catch
        [p,~,stats]=signrank(delta,0);
    end
    if isfield(stats,'signedrank'), statistic=stats.signedrank; else, statistic=NaN; end
end


function [lo,hi,prob]=bootstrapDifference(delta,nBootstrap)
    delta=delta(:); n=numel(delta); boot=zeros(nBootstrap,1);
    for b=1:nBootstrap, boot(b)=mean(delta(randi(n,n,1))); end
    ci=prctile(boot,[2.5 97.5]); lo=ci(1); hi=ci(2); prob=mean(boot<0);
end


function adjusted=holmAdjust(p)
    p=p(:); m=numel(p); [sp,order]=sort(p); a=zeros(m,1);
    for k=1:m, a(k)=min(1,(m-k+1)*sp(k)); end
    for k=2:m, a(k)=max(a(k),a(k-1)); end
    adjusted=zeros(m,1); adjusted(order)=a;
end


function makeRobustnessGridFigure(S,cleanSummary,faultTypes,outDir)
    datasets=["NUAA","PHM2010"]; models=["FullModel","AutoGate","OracleGate"];
    labels={'Full model','Automatic gate','Oracle gate'}; colors=[0 0.447 0.741;0.85 0.325 0.098;0.929 0.694 0.125];
    fig=figure('Visible','off','Color','w','Position',[50 50 1800 900]);
    tl=tiledlayout(fig,2,4,'TileSpacing','compact','Padding','compact');
    title(tl,'Multi-Sensor and Multi-Fault Robustness','FontWeight','bold','FontSize',16);
    for d=1:2
        cleanMAE=cleanSummary.MAE(cleanSummary.Dataset==datasets(d)&cleanSummary.Model=="FullClean");
        for f=1:4
            ax=nexttile(tl); hold(ax,'on');
            for m=1:3
                idx=S.Dataset==datasets(d)&S.FaultType==faultTypes(f)&S.Model==models(m);
                sub=S(idx,:); levels=unique(sub.SeverityLevel);
                values=zeros(numel(levels),1);
                for l=1:numel(levels)
                    values(l)=mean(sub.MAE(sub.SeverityLevel==levels(l)))/cleanMAE;
                end
                plot(ax,levels,values,'-o','Color',colors(m,:),'MarkerFaceColor',colors(m,:), ...
                    'LineWidth',1.6);
            end
            grid(ax,'on'); box(ax,'on'); xticks(ax,1:4);
            tickSub=S(S.Dataset==datasets(d)&S.FaultType==faultTypes(f),:);
            tickSub=sortrows(tickSub,'SeverityLevel');
            tickValues=zeros(4,1);
            for l=1:4, tickValues(l)=tickSub.SeverityValue(find(tickSub.SeverityLevel==l,1)); end
            if faultTypes(f)=="Saturation"||faultTypes(f)=="Dropout"
                xticklabels(ax,compose('%g%%',100*tickValues));
            else
                xticklabels(ax,compose('%g sigma',tickValues));
            end
            xlabel(ax,'Fault severity'); ylabel(ax,'MAE / clean-data MAE');
            title(ax,sprintf('%s: %s',datasets(d),faultTypes(f)));
            if d==1&&f==1, legend(ax,labels,'Location','northwest'); end
        end
    end
    exportgraphics(fig,fullfile(outDir,'multifault_robustness_grid.png'),'Resolution',300);
    exportgraphics(fig,fullfile(outDir,'multifault_robustness_grid.pdf'),'ContentType','vector'); close(fig);
end


function makeDetectionFigure(D,faultTypes,outDir)
    datasets=["NUAA","PHM2010"];
    fig=figure('Visible','off','Color','w','Position',[50 50 1800 900]);
    tl=tiledlayout(fig,2,4,'TileSpacing','compact','Padding','compact');
    title(tl,'Correct Faulty-Sensor Detection Rate','FontWeight','bold','FontSize',16);
    for d=1:2
        for f=1:4
            ax=nexttile(tl); sub=D(D.Dataset==datasets(d)&D.FaultType==faultTypes(f),:);
            sensors=unique(sub.FaultSensor,'stable'); M=NaN(numel(sensors),4);
            for s=1:numel(sensors), for l=1:4
                idx=sub.FaultSensor==sensors(s)&sub.SeverityLevel==l;
                M(s,l)=sub.CorrectSensorDetectionRate(idx);
            end, end
            imagesc(ax,M,[0 1]); colormap(ax,parula); colorbar(ax);
            xticks(ax,1:4); yticks(ax,1:numel(sensors)); yticklabels(ax,cellstr(sensors));
            xlabel(ax,'Severity level'); ylabel(ax,'Injected faulty sensor');
            title(ax,sprintf('%s: %s',datasets(d),faultTypes(f)));
            for s=1:size(M,1), for l=1:4
                text(ax,l,s,sprintf('%.2f',M(s,l)),'HorizontalAlignment','center', ...
                    'Color',chooseTextColor(M(s,l)),'FontSize',9);
            end, end
        end
    end
    exportgraphics(fig,fullfile(outDir,'multifault_detection_heatmap.png'),'Resolution',300);
    exportgraphics(fig,fullfile(outDir,'multifault_detection_heatmap.pdf'),'ContentType','vector'); close(fig);
end


function c=chooseTextColor(value)
    if value>0.55, c='k'; else, c='w'; end
end


function makeCasePairedFigure(C,statsTable,outDir)
    datasets=["NUAA","PHM2010"]; fig=figure('Visible','off','Color','w','Position',[100 100 1350 600]);
    tl=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    title(tl,'Experiment-Level Aggregate Robustness','FontWeight','bold','FontSize',16);
    improved=[0.2 0.55 0.8]; worsened=[0.85 0.33 0.1];
    for d=1:2
        ax=nexttile(tl); hold(ax,'on'); sub=C(C.Dataset==datasets(d),:);
        for i=1:height(sub)
            col=improved; if sub.AutoRelativeMAE(i)>sub.FullRelativeMAE(i), col=worsened; end
            plot(ax,[1 2],[sub.FullRelativeMAE(i),sub.AutoRelativeMAE(i)],'-o', ...
                'Color',col,'MarkerFaceColor',col,'LineWidth',1.2);
        end
        xlim(ax,[0.7 2.3]); xticks(ax,[1 2]); xticklabels(ax,{'Full model','Automatic gate'});
        ylabel(ax,'Mean corrupted MAE / clean MAE'); grid(ax,'on'); box(ax,'on');
        stat=statsTable(statsTable.Dataset==datasets(d),:);
        title(ax,sprintf('%s: n=%d experiments, Holm p=%.3g', ...
            datasets(d),height(sub),stat.HolmAdjustedP));
        h1=plot(ax,NaN,NaN,'-o','Color',improved,'MarkerFaceColor',improved);
        h2=plot(ax,NaN,NaN,'-o','Color',worsened,'MarkerFaceColor',worsened);
        legend(ax,[h1 h2],{'Robustness improved','Robustness worsened'},'Location','best');
    end
    exportgraphics(fig,fullfile(outDir,'multifault_case_paired.png'),'Resolution',300);
    exportgraphics(fig,fullfile(outDir,'multifault_case_paired.pdf'),'ContentType','vector'); close(fig);
end


function makeHeadlineFigure(S,D,C,outDir)
    datasets=["NUAA","PHM2010"]; models=["FullModel","AutoGate","OracleGate"];
    orderedFaults=unique(D.FaultType,'stable');
    fig=figure('Visible','off','Color','w','Position',[100 100 1450 650]);
    tl=tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
    title(tl,'Quality Gating across Sensors and Fault Modes','FontWeight','bold','FontSize',16);
    ax1=nexttile(tl); high=S(S.SeverityLevel==4,:); vals=zeros(3,2);
    for m=1:3, for d=1:2
        vals(m,d)=mean(high.NormalizedMAE(high.Model==models(m)&high.Dataset==datasets(d)));
    end, end
    bar(ax1,vals); grid(ax1,'on'); box(ax1,'on'); set(ax1,'XTickLabel',{'Full','Auto gate','Oracle gate'});
    ylabel(ax1,'Normalized MAE at highest severity'); title(ax1,'Prediction robustness');
    legend(ax1,cellstr(datasets),'Location','northwest');
    ax2=nexttile(tl); det=zeros(4,2);
    for f=1:4, for d=1:2
        idx=D.FaultType==orderedFaults(f)&D.Dataset==datasets(d)&D.SeverityLevel==4;
        det(f,d)=mean(D.CorrectSensorDetectionRate(idx));
    end, end
    bar(ax2,det); ylim(ax2,[0 1]); grid(ax2,'on'); box(ax2,'on');
    set(ax2,'XTickLabel',cellstr(orderedFaults),'XTickLabelRotation',20);
    ylabel(ax2,'Correct-sensor detection rate'); title(ax2,'Automatic quality diagnosis');
    legend(ax2,cellstr(datasets),'Location','southeast');
    ax3=nexttile(tl); paired=[C.FullRelativeMAE,C.AutoRelativeMAE,C.OracleRelativeMAE];
    values=paired(:); groups=repelem((1:3)',height(C));
    boxchart(ax3,groups,values); xticks(ax3,1:3);
    xticklabels(ax3,{'Full','Auto gate','Oracle gate'}); grid(ax3,'on'); box(ax3,'on');
    ylabel(ax3,'Experiment-level relative MAE'); title(ax3,'Across 12 independent experiments');
    exportgraphics(fig,fullfile(outDir,'multifault_headline_summary.png'),'Resolution',300);
    exportgraphics(fig,fullfile(outDir,'multifault_headline_summary.pdf'),'ContentType','vector'); close(fig);
end


function writeMultifaultReadme(outDir,faultTypes,severityMatrix,severityUnits,cleanSummary,statsTable)
    fid=fopen(fullfile(outDir,'README_multifault_experiment.txt'),'w','n','UTF-8');
    assert(fid>0,'Cannot create README.'); cleanup=onCleanup(@()fclose(fid));
    fprintf(fid,'MULTI-SENSOR MULTI-FAULT ROBUSTNESS EXPERIMENT\n');
    fprintf(fid,'MATLAB %s\n\n',version);
    fprintf(fid,'Independent units: NUAA W1-W9 (n=9) and PHM2010 c1/c4/c6 (n=3).\n');
    fprintf(fid,'Technical run-level observations were not treated as independent inferential n.\n');
    fprintf(fid,'Sensors: force_z, vibration_x, vibration_y.\n');
    fprintf(fid,'Faults and severity values:\n');
    for f=1:numel(faultTypes)
        fprintf(fid,'- %s: %s %s\n',faultTypes(f),mat2str(severityMatrix(f,:)),severityUnits(f));
    end
    fprintf(fid,'\nAutomatic detector training:\n');
    fprintf(fid,'Robust median/MAD ranges and the 99th percentile threshold were estimated using clean outer-training experiments only.\n');
    fprintf(fid,'The detector used sensor statistics plus rail, zero, flatline and difference-energy descriptors.\n');
    fprintf(fid,'OracleGate uses the injected fault identity only as an upper-bound comparison; it is not the proposed deployable result.\n');
    fprintf(fid,'\nClean-data results:\n');
    for i=1:height(cleanSummary)
        fprintf(fid,'%s %s: MAE=%.6f, R2=%.4f, clean false-alarm rate=%.4f.\n', ...
            cleanSummary.Dataset(i),cleanSummary.Model(i),cleanSummary.MAE(i), ...
            cleanSummary.R2(i),cleanSummary.CleanFalseAlarmRate(i));
    end
    fprintf(fid,'\nExperiment-level robustness statistics:\n');
    for i=1:height(statsTable)
        fprintf(fid,'%s: n=%d, full relative MAE=%.4f, automatic-gate relative MAE=%.4f, improvement=%.2f%%, raw p=%.6g, Holm p=%.6g, CI=[%.4f, %.4f].\n', ...
            statsTable.Dataset(i),statsTable.IndependentExperimentN(i), ...
            statsTable.FullRelativeMAE(i),statsTable.AutoRelativeMAE(i), ...
            statsTable.RelativeMAEImprovementPercent(i),statsTable.WilcoxonP(i), ...
            statsTable.HolmAdjustedP(i), ...
            statsTable.BootstrapCILower(i),statsTable.BootstrapCIUpper(i));
    end
end
