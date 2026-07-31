function [predictionAll,predictionQuality,predictionGated, ...
    bestLambdaAll,bestLambdaQuality,bestGateThreshold, ...
    gateValidationMAE] = groupedGatedRidgePrediction( ...
    XAll,XQuality,qualityRatio,y,groupID, ...
    outerTrainMask,outerTestMask,lambdaGrid,gateThresholdGrid)
%GROUPEDGATEDRIDGEPREDICTION Strict nested grouped quality-gated ridge.
% Lambda values and the gate threshold are selected only from the outer
% training cases. Standardization is performed by fitRidgeAndPredict using
% each corresponding training fold only.

% The threshold grid must be prespecified before examining outer-test
% outcomes. Ties are resolved in favor of the largest threshold, which
% makes the least aggressive switch to the quality-aware branch.

% Outputs predictionAll, predictionQuality, and predictionGated contain
% predictions for outerTestMask only.

outerTrainMask = logical(outerTrainMask(:));
outerTestMask = logical(outerTestMask(:));
qualityRatio = qualityRatio(:);
y = y(:);
groupID = groupID(:);
gateThresholdGrid = gateThresholdGrid(:)';

assert(~any(outerTrainMask & outerTestMask), ...
    "Outer training and test masks overlap.");
assert(all(isfinite(gateThresholdGrid)), ...
    "Gate threshold grid contains non-finite values.");

[predictionAll,bestLambdaAll] = groupedRidgePrediction( ...
    XAll,y,groupID,outerTrainMask,outerTestMask,lambdaGrid);

[predictionQuality,bestLambdaQuality] = groupedRidgePrediction( ...
    XQuality,y,groupID,outerTrainMask,outerTestMask,lambdaGrid);

trainingCases = unique(groupID(outerTrainMask));
innerPredictionAll = nan(size(y));
innerPredictionQuality = nan(size(y));

for validationIndex = 1:numel(trainingCases)
    validationCase = trainingCases(validationIndex);

    innerValidationMask = outerTrainMask & groupID == validationCase;
    innerTrainingMask = outerTrainMask & groupID ~= validationCase;

    innerPredictionAll(innerValidationMask) = fitRidgeAndPredict( ...
        XAll(innerTrainingMask,:),y(innerTrainingMask), ...
        XAll(innerValidationMask,:),bestLambdaAll);

    innerPredictionQuality(innerValidationMask) = fitRidgeAndPredict( ...
        XQuality(innerTrainingMask,:),y(innerTrainingMask), ...
        XQuality(innerValidationMask,:),bestLambdaQuality);
end

assert(all(isfinite(innerPredictionAll(outerTrainMask))), ...
    "Incomplete inner predictions for all-sensor branch.");
assert(all(isfinite(innerPredictionQuality(outerTrainMask))), ...
    "Incomplete inner predictions for quality-aware branch.");

gateValidationMAE = nan(1,numel(gateThresholdGrid));

for thresholdIndex = 1:numel(gateThresholdGrid)
    threshold = gateThresholdGrid(thresholdIndex);
    innerGatedPrediction = innerPredictionAll;
    switchMask = outerTrainMask & qualityRatio >= threshold;
    innerGatedPrediction(switchMask) = innerPredictionQuality(switchMask);

    gateValidationMAE(thresholdIndex) = mean(abs( ...
        innerGatedPrediction(outerTrainMask)-y(outerTrainMask)));
end

minimumMAE = min(gateValidationMAE);
tieTolerance = 1e-12 * max(1,abs(minimumMAE));
candidateIndices = find(gateValidationMAE <= minimumMAE + tieTolerance);
[~,largestThresholdPosition] = max(gateThresholdGrid(candidateIndices));
bestThresholdIndex = candidateIndices(largestThresholdPosition);
bestGateThreshold = gateThresholdGrid(bestThresholdIndex);

predictionGated = predictionAll;
outerQualityRatio = qualityRatio(outerTestMask);
outerSwitchMask = outerQualityRatio >= bestGateThreshold;
predictionGated(outerSwitchMask) = predictionQuality(outerSwitchMask);

end
