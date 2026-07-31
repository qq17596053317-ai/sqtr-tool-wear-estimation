clear; clc; close all;
paperRoot = fileparts(mfilename('fullpath'));
rng(20260717, 'twister');

dataDir = paperRoot;
outDir = fullfile(dataDir, 'external_validation_results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

logFile = fullfile(outDir, 'external_validation_matlab_log.txt');
if exist(logFile, 'file')
    delete(logFile);
end
diary(logFile);
cleanupDiary = onCleanup(@() diary('off'));

fprintf('MATLAB version: %s\n', version);
fprintf('External validation started: %s\n', char(datetime('now')));

nuaaFile = fullfile(dataDir, 'nuaa_orthogonal_bundle_high_resolution.csv');
phmFile = fullfile(dataDir, 'phm2010_bundle_high_resolution.csv');
assert(isfile(nuaaFile), 'NUAA file not found: %s', nuaaFile);
assert(isfile(phmFile), 'PHM2010 file not found: %s', phmFile);

corruptionLevels = [0 0.10 0.30 0.50 0.70 1.00];
lambdaGrid = 10.^(-4:0.5:4);
nBootstrap = 10000;

fprintf('\nReading and aggregating NUAA data...\n');
nuaa = buildRunLevelDataset(nuaaFile, "NUAA", corruptionLevels);
fprintf('NUAA: %d run-level observations from %d experiments.\n', ...
    height(nuaa.runTable), numel(unique(nuaa.experiment)));

fprintf('\nReading and aggregating PHM2010 data...\n');
phm = buildRunLevelDataset(phmFile, "PHM2010", corruptionLevels);
fprintf('PHM2010: %d run-level observations from %d experiments.\n', ...
    height(phm.runTable), numel(unique(phm.experiment)));

auditTable = [makeAuditTable(nuaa); makeAuditTable(phm)];
writetable(auditTable, fullfile(outDir, 'external_dataset_audit.csv'));
writetable(nuaa.featureTable, fullfile(outDir, 'nuaa_run_level_features.csv'));
writetable(phm.featureTable, fullfile(outDir, 'phm2010_run_level_features.csv'));

fprintf('\n===== Dataset audit =====\n');
disp(auditTable);

fprintf('\nRunning nested leave-one-experiment-out validation for NUAA...\n');
nuaaResult = nestedExternalValidation(nuaa, lambdaGrid, corruptionLevels);

fprintf('\nRunning nested leave-one-experiment-out validation for PHM2010...\n');
phmResult = nestedExternalValidation(phm, lambdaGrid, corruptionLevels);

summaryTable = [nuaaResult.summary; phmResult.summary];
caseTable = [nuaaResult.caseMetrics; phmResult.caseMetrics];
lambdaTable = [nuaaResult.lambdaTable; phmResult.lambdaTable];
predictionTable = [nuaaResult.predictions; phmResult.predictions];
robustnessTable = [nuaaResult.robustness; phmResult.robustness];
robustPredictionTable = [nuaaResult.robustPredictions; phmResult.robustPredictions];

statsTable = makeStatisticsTable(nuaaResult, phmResult, nBootstrap);

writetable(summaryTable, fullfile(outDir, 'external_validation_summary.csv'));
writetable(caseTable, fullfile(outDir, 'external_validation_case_metrics.csv'));
writetable(lambdaTable, fullfile(outDir, 'external_validation_selected_lambdas.csv'));
writetable(predictionTable, fullfile(outDir, 'external_validation_predictions.csv'));
writetable(robustnessTable, fullfile(outDir, 'external_saturation_robustness.csv'));
writetable(robustPredictionTable, fullfile(outDir, 'external_saturation_predictions.csv'));
writetable(statsTable, fullfile(outDir, 'external_validation_statistics.csv'));

fprintf('\n===== Clean external-validation summary =====\n');
disp(summaryTable);
fprintf('\n===== Paired experiment-level statistics =====\n');
disp(statsTable);
fprintf('\n===== Saturation robustness summary =====\n');
disp(robustnessTable);

makePredictionFigure(nuaaResult, phmResult, outDir);
makeModelComparisonFigure(summaryTable, outDir);
makeRobustnessFigure(robustnessTable, outDir);
makePairedCaseFigure(nuaaResult, phmResult, outDir);

save(fullfile(outDir, 'external_validation_complete_results.mat'), ...
    'nuaa', 'phm', 'nuaaResult', 'phmResult', 'auditTable', ...
    'summaryTable', 'caseTable', 'lambdaTable', 'predictionTable', ...
    'robustnessTable', 'robustPredictionTable', 'statsTable', ...
    'corruptionLevels', 'lambdaGrid', '-v7.3');

writeMethodsAndReadme(outDir, nuaa, phm, corruptionLevels, lambdaGrid, statsTable);

fprintf('\nAll external-validation files saved to:\n%s\n', outDir);
fprintf('External validation finished: %s\n', char(datetime('now')));


function D = buildRunLevelDataset(filePath, datasetName, corruptionLevels)
    T = readtable(filePath, 'VariableNamingRule', 'preserve');
    tag = string(T.experiment_tag);
    timestamp = double(T.timestamp);
    wear = double(T.tool_wear);

    if datasetName == "NUAA"
        sensorNames = ["force_z", "bending_moment_x", "bending_moment_y", ...
            "torsion", "vibration1", "vibration2", "spindle_power", "spindle_current"];
        processNames = ["feed_per_tooth", "spindle_speed", "axial_cutting_depth"];
        [groupId, ~, ~] = findgroups(tag, double(T.experiment_csv_n));
    else
        sensorNames = ["force_x", "force_y", "force_z", ...
            "vibration_x", "vibration_y", "vibration_z", "acoustic_emission_rms"];
        processNames = strings(1, 0);
        % PHM2010 run boundaries are encoded by the timestamp cadence in
        % the curated high-resolution bundle. Within a cut, consecutive
        % timestamps differ by about 0.04 s; between concatenated cuts the
        % increment is about 2e-5 s. Detect this cadence discontinuity
        % without using tool-wear labels.
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
                'Could not estimate timestamp cadence for %s.',tags(i));
            boundaryThreshold = 0.1 * typicalCadence;
            isNewRun = [true; positiveIncrement < boundaryThreshold];
            localRun = cumsum(isNewRun);
            groupId(rows) = nextGroup + localRun;
            nextGroup = nextGroup + max(localRun);
        end
    end

    assert(all(groupId > 0), 'Some rows were not assigned to a run group.');
    nGroups = max(groupId);
    statNames = ["Mean", "RMS", "Std", "PeakToPeak", "Skewness", ...
        "Kurtosis", "Median", "MAD", "CrestFactor"];
    nStats = numel(statNames);
    nSensors = numel(sensorNames);
    nFeatures = nSensors * nStats;
    nLevels = numel(corruptionLevels);

    featureNames = strings(1, nFeatures);
    for s = 1:nSensors
        block = (s-1)*nStats + (1:nStats);
        featureNames(block) = sensorNames(s) + "_" + statNames;
    end
    targetSensor = find(sensorNames == "force_z", 1);
    assert(~isempty(targetSensor), 'force_z must be available in both datasets.');
    targetFeatureIdx = (targetSensor-1)*nStats + (1:nStats);

    Xsensor = NaN(nGroups, nFeatures);
    XsensorCorrupt = NaN(nGroups, nFeatures, nLevels);
    Xprocess = NaN(nGroups, numel(processNames));
    experiment = strings(nGroups, 1);
    elapsedTime = NaN(nGroups, 1);
    actualWear = NaN(nGroups, 1);
    rawSampleCount = zeros(nGroups, 1);
    sourceGroup = zeros(nGroups, 1);

    tags = unique(tag, 'stable');
    expStart = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for i = 1:numel(tags)
        expStart(char(tags(i))) = min(timestamp(tag == tags(i)), [], 'omitnan');
    end

    for g = 1:nGroups
        rows = find(groupId == g);
        expTag = tag(rows(1));
        experiment(g) = expTag;
        rawSampleCount(g) = numel(rows);
        sourceGroup(g) = g;
        elapsedTime(g) = median(timestamp(rows), 'omitnan') - expStart(char(expTag));
        actualWear(g) = median(wear(rows), 'omitnan');

        cleanRow = NaN(1, nFeatures);
        corruptRows = NaN(nLevels, nFeatures);
        for s = 1:nSensors
            x = double(T.(char(sensorNames(s)))(rows));
            x = x(isfinite(x));
            block = (s-1)*nStats + (1:nStats);
            cleanFeatures = signalFeatures(x);
            cleanRow(block) = cleanFeatures;
            corruptRows(:, block) = repmat(cleanFeatures, nLevels, 1);
            if s == targetSensor
                for l = 1:nLevels
                    xc = injectUpperRailSaturation(x, corruptionLevels(l));
                    corruptRows(l, block) = signalFeatures(xc);
                end
            end
        end
        Xsensor(g, :) = cleanRow;
        for l = 1:nLevels
            XsensorCorrupt(g, :, l) = corruptRows(l, :);
        end
        for p = 1:numel(processNames)
            Xprocess(g, p) = median(double(T.(char(processNames(p)))(rows)), 'omitnan');
        end
    end

    sortKey = table(experiment, elapsedTime, sourceGroup, ...
        'VariableNames', {'Experiment','ElapsedTime','SourceGroup'});
    [~, order] = sortrows(sortKey, {'Experiment','ElapsedTime','SourceGroup'});
    experiment = experiment(order);
    elapsedTime = elapsedTime(order);
    actualWear = actualWear(order);
    rawSampleCount = rawSampleCount(order);
    sourceGroup = sourceGroup(order);
    Xsensor = Xsensor(order, :);
    XsensorCorrupt = XsensorCorrupt(order, :, :);
    Xprocess = Xprocess(order, :);

    runIndex = zeros(numel(order), 1);
    tags = unique(experiment, 'stable');
    for i = 1:numel(tags)
        idx = find(experiment == tags(i));
        runIndex(idx) = (1:numel(idx))';
    end
    Xtime = [elapsedTime, runIndex];

    datasetColumn = repmat(datasetName, numel(order), 1);
    runTable = table(datasetColumn, experiment, runIndex, sourceGroup, ...
        rawSampleCount, elapsedTime, actualWear, ...
        'VariableNames', {'Dataset','Experiment','RunIndex','SourceGroup', ...
        'RawSampleCount','ElapsedTime','ActualWear'});

    featureTable = runTable;
    for p = 1:numel(processNames)
        featureTable.(char(processNames(p))) = Xprocess(:, p);
    end
    for j = 1:nFeatures
        featureTable.(char(featureNames(j))) = Xsensor(:, j);
    end

    assert(all(isfinite(Xsensor), 'all'), 'Non-finite sensor features detected.');
    assert(all(isfinite(Xprocess), 'all'), 'Non-finite process features detected.');
    assert(all(isfinite(Xtime), 'all'), 'Non-finite time features detected.');
    assert(all(isfinite(actualWear)), 'Non-finite wear labels detected.');

    D.name = datasetName;
    D.runTable = runTable;
    D.featureTable = featureTable;
    D.sensorX = Xsensor;
    D.sensorXCorrupt = XsensorCorrupt;
    D.processX = Xprocess;
    D.timeX = Xtime;
    D.y = actualWear;
    D.experiment = experiment;
    D.sensorNames = sensorNames;
    D.processNames = processNames;
    D.sensorFeatureNames = featureNames;
    D.targetFeatureIdx = targetFeatureIdx;
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


function xc = injectUpperRailSaturation(x, ratio)
    xc = x(:);
    n = numel(xc);
    if n == 0 || ratio <= 0
        return;
    end
    k = min(n, max(1, round(ratio * n)));
    idx = unique(round(linspace(1, n, k)));
    upperRail = max(xc, [], 'omitnan');
    xc(idx) = upperRail;
end


function audit = makeAuditTable(D)
    cases = unique(D.experiment, 'stable');
    n = numel(cases);
    dataset = repmat(D.name, n, 1);
    rawSamples = zeros(n, 1);
    runCount = zeros(n, 1);
    timeMin = zeros(n, 1);
    timeMax = zeros(n, 1);
    wearMin = zeros(n, 1);
    wearMax = zeros(n, 1);
    spearmanTimeWear = NaN(n, 1);
    for i = 1:n
        idx = D.experiment == cases(i);
        rawSamples(i) = sum(D.runTable.RawSampleCount(idx));
        runCount(i) = sum(idx);
        timeMin(i) = min(D.runTable.ElapsedTime(idx));
        timeMax(i) = max(D.runTable.ElapsedTime(idx));
        wearMin(i) = min(D.y(idx));
        wearMax(i) = max(D.y(idx));
        if sum(idx) > 2
            spearmanTimeWear(i) = corr(D.runTable.ElapsedTime(idx), D.y(idx), ...
                'Type', 'Spearman', 'Rows', 'complete');
        end
    end
    audit = table(dataset, cases, rawSamples, runCount, timeMin, timeMax, ...
        wearMin, wearMax, spearmanTimeWear, ...
        'VariableNames', {'Dataset','Experiment','RawSamples','RunLevelN', ...
        'ElapsedTimeMin','ElapsedTimeMax','WearMin','WearMax','SpearmanTimeWear'});
end


function R = nestedExternalValidation(D, lambdaGrid, corruptionLevels)
    y = D.y;
    groups = D.experiment;
    cases = unique(groups, 'stable');
    n = numel(y);
    nLevels = numel(corruptionLevels);

    Xsensor = [D.processX, D.sensorX];
    Xtime = [D.processX, D.timeX];
    XsensorTime = [D.processX, D.sensorX, D.timeX];

    keepSensor = true(1, size(D.sensorX, 2));
    keepSensor(D.targetFeatureIdx) = false;
    XdropTime = [D.processX, D.sensorX(:, keepSensor), D.timeX];

    predMean = NaN(n, 1);
    predSensor = NaN(n, 1);
    predTime = NaN(n, 1);
    predSensorTime = NaN(n, 1);
    predCorruptFull = NaN(n, nLevels);
    predCorruptDrop = NaN(n, nLevels);
    predCorruptGated = NaN(n, nLevels);

    lambdaRows = cell(numel(cases), 6);
    for c = 1:numel(cases)
        testCase = cases(c);
        isTest = groups == testCase;
        isTrain = ~isTest;
        fprintf('  Outer test %s: training %d runs, testing %d runs\n', ...
            testCase, sum(isTrain), sum(isTest));

        lambdaSensor = groupedTuneLambda(Xsensor(isTrain, :), y(isTrain), ...
            groups(isTrain), lambdaGrid);
        lambdaTime = groupedTuneLambda(Xtime(isTrain, :), y(isTrain), ...
            groups(isTrain), lambdaGrid);
        lambdaFull = groupedTuneLambda(XsensorTime(isTrain, :), y(isTrain), ...
            groups(isTrain), lambdaGrid);
        lambdaDrop = groupedTuneLambda(XdropTime(isTrain, :), y(isTrain), ...
            groups(isTrain), lambdaGrid);

        predMean(isTest) = mean(y(isTrain));
        modelSensor = fitRidge(Xsensor(isTrain, :), y(isTrain), lambdaSensor);
        modelTime = fitRidge(Xtime(isTrain, :), y(isTrain), lambdaTime);
        modelFull = fitRidge(XsensorTime(isTrain, :), y(isTrain), lambdaFull);
        modelDrop = fitRidge(XdropTime(isTrain, :), y(isTrain), lambdaDrop);

        predSensor(isTest) = predictRidge(modelSensor, Xsensor(isTest, :));
        predTime(isTest) = predictRidge(modelTime, Xtime(isTest, :));
        predSensorTime(isTest) = predictRidge(modelFull, XsensorTime(isTest, :));

        for l = 1:nLevels
            corruptSensors = squeeze(D.sensorXCorrupt(isTest, :, l));
            if sum(isTest) == 1
                corruptSensors = reshape(corruptSensors, 1, []);
            end
            XcorruptFull = [D.processX(isTest, :), corruptSensors, D.timeX(isTest, :)];
            XcorruptDrop = [D.processX(isTest, :), corruptSensors(:, keepSensor), D.timeX(isTest, :)];
            fullPrediction = predictRidge(modelFull, XcorruptFull);
            dropPrediction = predictRidge(modelDrop, XcorruptDrop);
            predCorruptFull(isTest, l) = fullPrediction;
            predCorruptDrop(isTest, l) = dropPrediction;
            if corruptionLevels(l) >= 0.01
                predCorruptGated(isTest, l) = dropPrediction;
            else
                predCorruptGated(isTest, l) = fullPrediction;
            end
        end

        lambdaRows(c, :) = {D.name, testCase, lambdaSensor, lambdaTime, lambdaFull, lambdaDrop};
    end

    modelNames = ["TrainingMean"; "SensorRidge"; "TimeRidge"; "SensorTimeRidge"];
    predictions = [predMean, predSensor, predTime, predSensorTime];
    summary = makeSummary(D.name, y, predictions, modelNames, groups);
    caseMetrics = makeCaseMetrics(D.name, y, predictions, modelNames, groups);

    predictionTable = D.runTable;
    predictionTable.TrainingMean = predMean;
    predictionTable.SensorRidge = predSensor;
    predictionTable.TimeRidge = predTime;
    predictionTable.SensorTimeRidge = predSensorTime;

    lambdaTable = cell2table(lambdaRows, 'VariableNames', ...
        {'Dataset','TestExperiment','SensorLambda','TimeLambda','SensorTimeLambda','DropForceZTimeLambda'});
    lambdaTable.Dataset = string(lambdaTable.Dataset);
    lambdaTable.TestExperiment = string(lambdaTable.TestExperiment);

    robustNames = ["AllSensorTime"; "DropForceZTime"; "QualityGatedTime"];
    robustRows = cell(nLevels * numel(robustNames), 9);
    robustPredRows = cell(n * nLevels, 10);
    rr = 0;
    rp = 0;
    for l = 1:nLevels
        levelPred = [predCorruptFull(:, l), predCorruptDrop(:, l), predCorruptGated(:, l)];
        for m = 1:numel(robustNames)
            rr = rr + 1;
            metrics = regressionMetrics(y, levelPred(:, m), groups);
            robustRows(rr, :) = {D.name, corruptionLevels(l), 100*corruptionLevels(l), ...
                robustNames(m), metrics.MAE, metrics.RMSE, metrics.R2, ...
                metrics.MacroMAE, metrics.P95AbsoluteError};
        end
        for i = 1:n
            rp = rp + 1;
            robustPredRows(rp, :) = {D.name, groups(i), D.runTable.RunIndex(i), ...
                D.runTable.ElapsedTime(i), y(i), corruptionLevels(l), ...
                predCorruptFull(i,l), predCorruptDrop(i,l), ...
                predCorruptGated(i,l), D.runTable.RawSampleCount(i)};
        end
    end
    robustness = cell2table(robustRows, 'VariableNames', ...
        {'Dataset','CorruptionRatio','CorruptionPercent','Model','MAE','RMSE','R2','MacroMAE','P95AbsoluteError'});
    robustness.Dataset = string(robustness.Dataset);
    robustness.Model = string(robustness.Model);
    robustPredictions = cell2table(robustPredRows, 'VariableNames', ...
        {'Dataset','Experiment','RunIndex','ElapsedTime','ActualWear','CorruptionRatio', ...
        'AllSensorTimePrediction','DropForceZTimePrediction','QualityGatedTimePrediction','RawSampleCount'});
    robustPredictions.Dataset = string(robustPredictions.Dataset);
    robustPredictions.Experiment = string(robustPredictions.Experiment);

    R.name = D.name;
    R.y = y;
    R.groups = groups;
    R.modelNames = modelNames;
    R.predictionMatrix = predictions;
    R.summary = summary;
    R.caseMetrics = caseMetrics;
    R.lambdaTable = lambdaTable;
    R.predictions = predictionTable;
    R.robustness = robustness;
    R.robustPredictions = robustPredictions;
end


function lambda = groupedTuneLambda(X, y, groups, lambdaGrid)
    innerCases = unique(groups, 'stable');
    if numel(innerCases) < 2
        lambda = 1;
        return;
    end
    scores = NaN(numel(lambdaGrid), 1);
    for k = 1:numel(lambdaGrid)
        foldMAE = NaN(numel(innerCases), 1);
        for f = 1:numel(innerCases)
            isValidation = groups == innerCases(f);
            isTraining = ~isValidation;
            if sum(isTraining) < 2 || ~any(isValidation)
                continue;
            end
            model = fitRidge(X(isTraining, :), y(isTraining), lambdaGrid(k));
            yhat = predictRidge(model, X(isValidation, :));
            foldMAE(f) = mean(abs(yhat - y(isValidation)));
        end
        scores(k) = mean(foldMAE, 'omitnan');
    end
    [~, best] = min(scores);
    if isempty(best) || ~isfinite(scores(best))
        lambda = 1;
    else
        lambda = lambdaGrid(best);
    end
end


function model = fitRidge(X, y, lambda)
    X = double(X);
    y = double(y(:));
    muX = mean(X, 1, 'omitnan');
    muX(~isfinite(muX)) = 0;
    for j = 1:size(X, 2)
        bad = ~isfinite(X(:, j));
        X(bad, j) = muX(j);
    end
    sigmaX = std(X, 0, 1);
    sigmaX(~isfinite(sigmaX) | sigmaX < 1e-12) = 1;
    Xz = (X - muX) ./ sigmaX;
    muY = mean(y);
    yc = y - muY;
    p = size(Xz, 2);
    if p == 0
        beta = zeros(0, 1);
    else
        beta = (Xz' * Xz + lambda * eye(p)) \ (Xz' * yc);
    end
    model.muX = muX;
    model.sigmaX = sigmaX;
    model.muY = muY;
    model.beta = beta;
    model.lambda = lambda;
end


function yhat = predictRidge(model, X)
    X = double(X);
    for j = 1:size(X, 2)
        bad = ~isfinite(X(:, j));
        X(bad, j) = model.muX(j);
    end
    Xz = (X - model.muX) ./ model.sigmaX;
    yhat = model.muY + Xz * model.beta;
end


function summary = makeSummary(datasetName, y, predictions, modelNames, groups)
    nModels = numel(modelNames);
    dataset = repmat(datasetName, nModels, 1);
    MAE = zeros(nModels, 1);
    RMSE = zeros(nModels, 1);
    R2 = zeros(nModels, 1);
    MacroMAE = zeros(nModels, 1);
    P95AbsoluteError = zeros(nModels, 1);
    MaximumAbsoluteError = zeros(nModels, 1);
    for m = 1:nModels
        metrics = regressionMetrics(y, predictions(:, m), groups);
        MAE(m) = metrics.MAE;
        RMSE(m) = metrics.RMSE;
        R2(m) = metrics.R2;
        MacroMAE(m) = metrics.MacroMAE;
        P95AbsoluteError(m) = metrics.P95AbsoluteError;
        MaximumAbsoluteError(m) = metrics.MaximumAbsoluteError;
    end
    summary = table(dataset, modelNames, MAE, RMSE, R2, MacroMAE, ...
        P95AbsoluteError, MaximumAbsoluteError, ...
        'VariableNames', {'Dataset','Model','MAE','RMSE','R2','MacroMAE', ...
        'P95AbsoluteError','MaximumAbsoluteError'});
end


function caseMetrics = makeCaseMetrics(datasetName, y, predictions, modelNames, groups)
    cases = unique(groups, 'stable');
    rows = cell(numel(cases) * numel(modelNames), 6);
    r = 0;
    for c = 1:numel(cases)
        idx = groups == cases(c);
        for m = 1:numel(modelNames)
            r = r + 1;
            residual = predictions(idx, m) - y(idx);
            rows(r, :) = {datasetName, cases(c), sum(idx), modelNames(m), ...
                mean(abs(residual)), sqrt(mean(residual.^2))};
        end
    end
    caseMetrics = cell2table(rows, 'VariableNames', ...
        {'Dataset','Experiment','RunLevelN','Model','MAE','RMSE'});
    caseMetrics.Dataset = string(caseMetrics.Dataset);
    caseMetrics.Experiment = string(caseMetrics.Experiment);
    caseMetrics.Model = string(caseMetrics.Model);
end


function metrics = regressionMetrics(y, yhat, groups)
    residual = yhat - y;
    absError = abs(residual);
    metrics.MAE = mean(absError);
    metrics.RMSE = sqrt(mean(residual.^2));
    denominator = sum((y - mean(y)).^2);
    if denominator > 0
        metrics.R2 = 1 - sum(residual.^2) / denominator;
    else
        metrics.R2 = NaN;
    end
    cases = unique(groups, 'stable');
    caseMAE = zeros(numel(cases), 1);
    for c = 1:numel(cases)
        idx = groups == cases(c);
        caseMAE(c) = mean(absError(idx));
    end
    metrics.MacroMAE = mean(caseMAE);
    metrics.P95AbsoluteError = prctile(absError, 95);
    metrics.MaximumAbsoluteError = max(absError);
end


function statsTable = makeStatisticsTable(nuaaResult, phmResult, nBootstrap)
    datasetResults = {nuaaResult, phmResult};
    rows = cell(3, 13);
    allSensorCase = [];
    allTimeCase = [];
    allDelta = [];
    rawP = NaN(2, 1);
    for d = 1:2
        R = datasetResults{d};
        sensorCase = caseMAEVector(R, "SensorRidge");
        timeCase = caseMAEVector(R, "SensorTimeRidge");
        delta = timeCase - sensorCase;
        [p, signedRank] = safeSignrank(delta);
        rawP(d) = p;
        [ciLow, ciHigh, probabilityImprovement] = bootstrapMeanDifference(delta, nBootstrap);
        improvement = 100 * (mean(sensorCase) - mean(timeCase)) / mean(sensorCase);
        rows(d, 1:12) = {R.name, numel(delta), mean(sensorCase), mean(timeCase), ...
            mean(delta), improvement, sum(delta < 0), sum(delta > 0), ...
            p, signedRank, ciLow, ciHigh};
        allSensorCase = [allSensorCase; sensorCase]; %#ok<AGROW>
        allTimeCase = [allTimeCase; timeCase]; %#ok<AGROW>
        allDelta = [allDelta; delta]; %#ok<AGROW>
        rows{d, 13} = probabilityImprovement;
    end
    adjusted = holmAdjust(rawP);

    [pCombined, signedRankCombined] = safeSignrank(allDelta);
    [ciLow, ciHigh, probabilityImprovement] = bootstrapMeanDifference(allDelta, nBootstrap);
    improvement = 100 * (mean(allSensorCase) - mean(allTimeCase)) / mean(allSensorCase);
    rows(3, 1:13) = {"Combined", numel(allDelta), mean(allSensorCase), mean(allTimeCase), ...
        mean(allDelta), improvement, sum(allDelta < 0), sum(allDelta > 0), ...
        pCombined, signedRankCombined, ciLow, ciHigh, probabilityImprovement};

    statsTable = cell2table(rows, 'VariableNames', ...
        {'Dataset','IndependentExperimentN','SensorMacroMAE','SensorTimeMacroMAE', ...
        'MeanPairedDifference','MacroMAEImprovementPercent','ExperimentsImproved', ...
        'ExperimentsWorsened','WilcoxonP','SignedRankStatistic', ...
        'BootstrapCILower','BootstrapCIUpper','BootstrapProbabilityImprovement'});
    statsTable.Dataset = string(statsTable.Dataset);
    statsTable.HolmAdjustedP = [adjusted; NaN];
    statsTable.SignificantAfterHolm = statsTable.HolmAdjustedP < 0.05;
end


function values = caseMAEVector(R, modelName)
    T = R.caseMetrics;
    values = T.MAE(T.Model == modelName);
end


function [p, signedRank] = safeSignrank(delta)
    delta = delta(isfinite(delta) & delta ~= 0);
    if isempty(delta)
        p = 1;
        signedRank = 0;
        return;
    end
    try
        [p, ~, stats] = signrank(delta, 0, 'method', 'exact');
    catch
        [p, ~, stats] = signrank(delta, 0);
    end
    if isfield(stats, 'signedrank')
        signedRank = stats.signedrank;
    else
        signedRank = NaN;
    end
end


function [ciLow, ciHigh, probabilityImprovement] = bootstrapMeanDifference(delta, nBootstrap)
    delta = delta(:);
    n = numel(delta);
    bootstrapMeans = zeros(nBootstrap, 1);
    for b = 1:nBootstrap
        idx = randi(n, n, 1);
        bootstrapMeans(b) = mean(delta(idx));
    end
    ci = prctile(bootstrapMeans, [2.5 97.5]);
    ciLow = ci(1);
    ciHigh = ci(2);
    probabilityImprovement = mean(bootstrapMeans < 0);
end


function adjusted = holmAdjust(p)
    p = p(:);
    m = numel(p);
    [sortedP, order] = sort(p);
    adjustedSorted = zeros(m, 1);
    for k = 1:m
        adjustedSorted(k) = min(1, (m - k + 1) * sortedP(k));
    end
    for k = 2:m
        adjustedSorted(k) = max(adjustedSorted(k), adjustedSorted(k-1));
    end
    adjusted = zeros(m, 1);
    adjusted(order) = adjustedSorted;
end


function makePredictionFigure(nuaaResult, phmResult, outDir)
    results = {nuaaResult, phmResult};
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1500 1100]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'External Validation under Leave-One-Experiment-Out Evaluation', ...
        'FontWeight', 'bold', 'FontSize', 15);
    for d = 1:2
        R = results{d};
        [caseNames, ~, caseCodes] = unique(R.groups, 'stable');
        modelIndices = [2 4];
        names = ["Sensor ridge", "Sensor + operational time ridge"];
        for j = 1:2
            ax = nexttile(tl);
            scatter(ax, R.y, R.predictionMatrix(:, modelIndices(j)), 18, ...
                caseCodes, 'filled', 'MarkerFaceAlpha', 0.65);
            hold(ax, 'on');
            bounds = [min([R.y; R.predictionMatrix(:,modelIndices(j))]), ...
                max([R.y; R.predictionMatrix(:,modelIndices(j))])];
            padding = max(0.005, 0.05 * range(bounds));
            bounds = bounds + [-padding padding];
            plot(ax, bounds, bounds, 'k--', 'LineWidth', 1.2);
            xlim(ax, bounds); ylim(ax, bounds);
            grid(ax, 'on'); box(ax, 'on'); axis(ax, 'square');
            metrics = regressionMetrics(R.y, R.predictionMatrix(:,modelIndices(j)), R.groups);
            title(ax, sprintf('%s: %s\nMAE = %.4f, R^2 = %.3f', ...
                R.name, names(j), metrics.MAE, metrics.R2));
            xlabel(ax, 'Actual tool wear (mm)');
            ylabel(ax, 'Predicted tool wear (mm)');
            colormap(ax, parula(numel(caseNames)));
            clim(ax, [0.5, numel(caseNames) + 0.5]);
            cb = colorbar(ax);
            cb.Ticks = 1:numel(caseNames);
            cb.TickLabels = cellstr(caseNames);
            cb.Label.String = 'Held-out experiment';
        end
    end
    exportgraphics(fig, fullfile(outDir, 'external_validation_predictions.png'), 'Resolution', 300);
    exportgraphics(fig, fullfile(outDir, 'external_validation_predictions.pdf'), 'ContentType', 'vector');
    close(fig);
end


function makeModelComparisonFigure(summaryTable, outDir)
    models = ["TrainingMean", "SensorRidge", "TimeRidge", "SensorTimeRidge"];
    datasets = ["NUAA", "PHM2010"];
    displayNames = {'Training mean','Sensor ridge','Operational time ridge','Sensor + time ridge'};
    MAE = zeros(numel(models), numel(datasets));
    RMSE = zeros(numel(models), numel(datasets));
    for d = 1:numel(datasets)
        for m = 1:numel(models)
            idx = summaryTable.Dataset == datasets(d) & summaryTable.Model == models(m);
            MAE(m,d) = summaryTable.MAE(idx);
            RMSE(m,d) = summaryTable.RMSE(idx);
        end
    end
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1450 600]);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'External Dataset Model Comparison', 'FontWeight', 'bold', 'FontSize', 15);
    ax1 = nexttile(tl);
    bar(ax1, MAE, 'grouped'); grid(ax1, 'on'); box(ax1, 'on');
    set(ax1, 'XTickLabel', displayNames, 'XTickLabelRotation', 18);
    ylabel(ax1, 'MAE (mm)'); title(ax1, 'Mean absolute error');
    legend(ax1, cellstr(datasets), 'Location', 'northwest');
    ax2 = nexttile(tl);
    bar(ax2, RMSE, 'grouped'); grid(ax2, 'on'); box(ax2, 'on');
    set(ax2, 'XTickLabel', displayNames, 'XTickLabelRotation', 18);
    ylabel(ax2, 'RMSE (mm)'); title(ax2, 'Root mean squared error');
    legend(ax2, cellstr(datasets), 'Location', 'northwest');
    exportgraphics(fig, fullfile(outDir, 'external_validation_model_comparison.png'), 'Resolution', 300);
    exportgraphics(fig, fullfile(outDir, 'external_validation_model_comparison.pdf'), 'ContentType', 'vector');
    close(fig);
end


function makeRobustnessFigure(robustnessTable, outDir)
    datasets = ["NUAA", "PHM2010"];
    models = ["AllSensorTime", "DropForceZTime", "QualityGatedTime"];
    labels = {'All sensors + time','Drop force_z + time','Quality-gated + time'};
    colors = lines(3);
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1450 1000]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'Robustness to Artificial force_z Saturation', 'FontWeight', 'bold', 'FontSize', 15);
    for d = 1:2
        for metricIndex = 1:2
            ax = nexttile(tl);
            hold(ax, 'on');
            for m = 1:3
                idx = robustnessTable.Dataset == datasets(d) & robustnessTable.Model == models(m);
                sub = sortrows(robustnessTable(idx,:), 'CorruptionPercent');
                if metricIndex == 1
                    values = sub.MAE;
                else
                    values = sub.R2;
                end
                plot(ax, sub.CorruptionPercent, values, '-o', 'LineWidth', 1.8, ...
                    'Color', colors(m,:), 'MarkerFaceColor', colors(m,:));
            end
            grid(ax, 'on'); box(ax, 'on');
            xlabel(ax, 'Injected upper-rail saturation (%)');
            if metricIndex == 1
                ylabel(ax, 'MAE (mm)'); metricTitle = 'MAE';
            else
                ylabel(ax, 'R^2'); metricTitle = 'R^2';
            end
            title(ax, sprintf('%s: %s under corruption', datasets(d), metricTitle));
            legend(ax, labels, 'Location', 'best');
        end
    end
    exportgraphics(fig, fullfile(outDir, 'external_saturation_robustness.png'), 'Resolution', 300);
    exportgraphics(fig, fullfile(outDir, 'external_saturation_robustness.pdf'), 'ContentType', 'vector');
    close(fig);
end


function makePairedCaseFigure(nuaaResult, phmResult, outDir)
    results = {nuaaResult, phmResult};
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1300 580]);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'Experiment-Level Paired Comparison', 'FontWeight', 'bold', 'FontSize', 15);
    for d = 1:2
        R = results{d};
        sensorCase = caseMAEVector(R, "SensorRidge");
        timeCase = caseMAEVector(R, "SensorTimeRidge");
        ax = nexttile(tl); hold(ax, 'on');
        improvedColor = [0.2 0.55 0.8];
        worsenedColor = [0.85 0.33 0.1];
        for i = 1:numel(sensorCase)
            color = improvedColor;
            if timeCase(i) > sensorCase(i)
                color = worsenedColor;
            end
            plot(ax, [1 2], [sensorCase(i), timeCase(i)], '-o', ...
                'Color', color, 'LineWidth', 1.2, 'MarkerFaceColor', color);
        end
        xlim(ax, [0.7 2.3]); xticks(ax, [1 2]);
        xticklabels(ax, {'Sensor ridge','Sensor + time ridge'});
        ylabel(ax, 'Experiment-level MAE (mm)');
        title(ax, sprintf('%s (n = %d independent experiments)', R.name, numel(sensorCase)));
        grid(ax, 'on'); box(ax, 'on');
        hImproved = plot(ax, NaN, NaN, '-o', 'Color', improvedColor, ...
            'MarkerFaceColor', improvedColor, 'LineWidth', 1.2);
        hWorsened = plot(ax, NaN, NaN, '-o', 'Color', worsenedColor, ...
            'MarkerFaceColor', worsenedColor, 'LineWidth', 1.2);
        legend(ax, [hImproved hWorsened], {'MAE improved','MAE worsened'}, ...
            'Location', 'best');
    end
    exportgraphics(fig, fullfile(outDir, 'external_validation_paired_experiments.png'), 'Resolution', 300);
    exportgraphics(fig, fullfile(outDir, 'external_validation_paired_experiments.pdf'), 'ContentType', 'vector');
    close(fig);
end


function writeMethodsAndReadme(outDir, nuaa, phm, corruptionLevels, lambdaGrid, statsTable)
    filePath = fullfile(outDir, 'README_external_validation.txt');
    fid = fopen(filePath, 'w', 'n', 'UTF-8');
    assert(fid > 0, 'Cannot create README file.');
    cleanupFile = onCleanup(@() fclose(fid));
    fprintf(fid, 'EXTERNAL VALIDATION: NUAA AND PHM2010\n');
    fprintf(fid, 'Generated with MATLAB %s\n\n', version);
    fprintf(fid, 'Design\n');
    fprintf(fid, '- NUAA independent units: W1-W9 experimental sequences (n=%d).\n', numel(unique(nuaa.experiment)));
    fprintf(fid, '- PHM2010 independent units: c1, c4 and c6 cutter sequences (n=%d).\n', numel(unique(phm.experiment)));
    fprintf(fid, '- Raw high-frequency rows were treated as technical subsamples and aggregated to cutting-run level.\n');
    fprintf(fid, '- PHM2010 cutting-run boundaries were detected from timestamp-cadence discontinuities; wear labels were not used for segmentation.\n');
    fprintf(fid, '- Outer validation: leave one entire experiment/cutter sequence out.\n');
    fprintf(fid, '- Hyperparameter selection: inner leave-one-training-experiment-out validation.\n');
    fprintf(fid, '- Lambda grid: %s.\n', mat2str(lambdaGrid));
    fprintf(fid, '- Sensor features: mean, RMS, standard deviation, peak-to-peak, skewness, kurtosis, median, MAD and crest factor.\n');
    fprintf(fid, '- Operational-history features: elapsed machining time and run index; neither uses the wear label.\n');
    fprintf(fid, '- No held-out experiment labels were used for preprocessing, tuning or model fitting.\n\n');
    fprintf(fid, 'Quality-gating stress test\n');
    fprintf(fid, '- Common channel: force_z.\n');
    fprintf(fid, '- Injected upper-rail ratios: %s.\n', mat2str(corruptionLevels));
    fprintf(fid, '- Gate threshold: 1%%. At or above the threshold, prediction switches to a branch trained without force_z.\n\n');
    fprintf(fid, 'Statistical unit\n');
    fprintf(fid, '- Inferential n is the number of independent experiment/cutter sequences, not the number of raw samples or run-level windows.\n');
    fprintf(fid, '- Paired Wilcoxon signed-rank tests compare experiment-level MAE. Two dataset-specific tests use Holm adjustment.\n');
    fprintf(fid, '- Bootstrap confidence intervals resample independent experiment-level paired differences.\n\n');
    fprintf(fid, 'Important limitations\n');
    fprintf(fid, '- PHM2010 contains only three independent cutter sequences, so dataset-specific statistical power is low.\n');
    fprintf(fid, '- Wear is strongly monotonic with elapsed time in both curated bundles. Results support predictive validity under these benchmark conditions, not a causal wear mechanism.\n');
    fprintf(fid, '- This is dataset-specific retraining under identical methodology, not zero-shot transfer of NASA model coefficients across incompatible sensor units.\n\n');
    fprintf(fid, 'Statistics summary\n');
    for i = 1:height(statsTable)
        fprintf(fid, '%s: n=%d, sensor macro-MAE=%.6f, sensor+time macro-MAE=%.6f, delta=%.6f, p=%.6g, bootstrap 95%% CI=[%.6f, %.6f].\n', ...
            statsTable.Dataset(i), statsTable.IndependentExperimentN(i), ...
            statsTable.SensorMacroMAE(i), statsTable.SensorTimeMacroMAE(i), ...
            statsTable.MeanPairedDifference(i), statsTable.WilcoxonP(i), ...
            statsTable.BootstrapCILower(i), statsTable.BootstrapCIUpper(i));
    end
end
