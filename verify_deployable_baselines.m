clear; clc;

rootDir=fileparts(mfilename('fullpath'));
resultDir=fullfile(rootDir,'additional_validation_results','reviewer_improvements');
unitFile=fullfile(resultDir,'random_fault_experiment_means.csv');
statsFile=fullfile(resultDir,'deployable_baseline_statistics.csv');
hyperFile=fullfile(resultDir,'mask_dropout_selected_hyperparameters.csv');

assert(isfile(unitFile),'Missing unit-level deployable-baseline results.');
assert(isfile(statsFile),'Missing deployable-baseline statistical results.');
assert(isfile(hyperFile),'Missing selected mask-dropout hyperparameters.');

U=readtable(unitFile,'VariableNamingRule','preserve');
S=readtable(statsFile,'VariableNamingRule','preserve');
H=readtable(hyperFile,'VariableNamingRule','preserve');

assert(height(U)==12,'Expected 12 independent-unit rows.');
assert(height(S)==2,'Expected two deployable-baseline contrasts.');
assert(height(H)==12,'Expected one selected mask-dropout policy per outer unit.');
assert(all(U.TechnicalRepeatN==30),'Technical repeat count must be 30.');

expected=[1.39226698409099,1.05825011004593,1.06645988958118,1.08267571551188];
observed=[mean(U.FullRelativeMAE),mean(U.MedianReplacementRelativeMAE), ...
    mean(U.MaskDropoutRelativeMAE),mean(U.AutoRelativeMAE)];
assert(max(abs(observed-expected))<1e-10,'Archived method means changed unexpectedly.');

expectedDifference=[-0.0244256054659535;-0.0162158259307023];
assert(max(abs(S.MeanDifferenceComparatorMinusSQTR-expectedDifference))<1e-10, ...
    'Archived comparator-minus-SQTR differences changed unexpectedly.');
assert(all(abs(S.HolmAdjustedP-1)<1e-12),'Expected Holm-adjusted p values of 1.');
assert(all(ismember(H.SelectedDropoutRate,[0.25 0.50 1.00])), ...
    'Selected dropout rate fell outside the prespecified grid.');

fprintf('Deployable-baseline archive verified: 12 independent units, 30 technical repeats.\n');
disp(S);
