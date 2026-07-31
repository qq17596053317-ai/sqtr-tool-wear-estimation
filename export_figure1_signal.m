function export_figure1_signal(inputMat, outputCsv)
% Export two objectively selected smcDC records for Python-only figure rendering.

raw = load(inputMat, "mill");
mill = raw.mill;

healthyIndex = 1;
saturatedIndex = 8;
healthy = double(mill(healthyIndex).smcDC(:));
saturated = double(mill(saturatedIndex).smcDC(:));

if numel(healthy) ~= numel(saturated)
    error("Selected records have unequal signal lengths.");
end

sampleIndex = (1:numel(healthy))';
normalizedPosition = 100 .* (sampleIndex - 1) ./ max(numel(healthy) - 1, 1);
source = table(sampleIndex, normalizedPosition, healthy, saturated, ...
    'VariableNames', {'SampleIndex','NormalizedPositionPercent', ...
    'HealthySmcDC','SaturatedSmcDC'});
writetable(source, outputCsv);
end
