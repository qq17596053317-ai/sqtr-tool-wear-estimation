clear;
clc;

projectFolder = fileparts(mfilename("fullpath"));
outputFolder = fullfile(projectFolder,"audit_outputs");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

%% NASA elapsed-time ablation: independent-case estimand
timeCases = readtable(fullfile(projectFolder,"mill","results", ...
    "time_model_cases.csv"),VariableNamingRule="preserve");
timeDifference = timeCases.RidgeGatedMAE - timeCases.TimeGatedMAE;
timeCases.NoTimeMinusSQTR_MAE_mm = timeDifference;
assert(height(timeCases) == 16,"Expected 16 NASA cases.");
assert(sum(timeDifference > 0) == 11,"Expected 11/16 cases to worsen without time.");

stream = RandStream("mt19937ar",Seed=20260730);
bootstrapMeans = zeros(10000,1);
for b = 1:numel(bootstrapMeans)
    selected = randi(stream,height(timeCases),height(timeCases),1);
    bootstrapMeans(b) = mean(timeDifference(selected));
end
timeCI = prctile(bootstrapMeans,[2.5 97.5]);
[timeP,~,timeStats] = signrank(timeCases.RidgeGatedMAE, ...
    timeCases.TimeGatedMAE,Method="exact");

timeAudit = table(mean(timeDifference),median(timeDifference),timeCI(1), ...
    timeCI(2),sum(timeDifference > 0),sum(timeDifference < 0), ...
    height(timeCases),timeP,timeStats.signedrank,20260730,10000, ...
    VariableNames=["MeanCaseMacroDifferenceNoTimeMinusSQTR_mm", ...
    "MedianCaseDifferenceNoTimeMinusSQTR_mm","BootstrapCILower_mm", ...
    "BootstrapCIUpper_mm","CasesWorseWithoutTime","CasesBetterWithoutTime", ...
    "IndependentCaseN","ExactWilcoxonP","SignedRankStatistic", ...
    "BootstrapSeed","BootstrapIterations"]);
writetable(timeAudit,fullfile(outputFolder,"time_ablation_case_macro_audit.csv"));
writetable(timeCases,fullfile(outputFolder,"time_ablation_case_results.csv"));

%% Input-matched routing comparison
route = readtable(fullfile(projectFolder,"targeted_reviewer_experiments", ...
    "nasa_qi_role_ablation","qi_role_route_comparisons.csv"), ...
    VariableNamingRule="preserve",TextType="string");
route = route(route.Comparison == "RouteOnly_vs_AllTime",:);
assert(height(route) == 1,"Input-matched route comparison missing.");
allTimePooled = 0.116210316365672;
sqtrPooled = 0.108773104678773;
routeReduction = 100 * (allTimePooled - sqtrPooled) / allTimePooled;
assert(abs(route.PooledMAEDifferenceBMinusA - (sqtrPooled-allTimePooled)) < 1e-12);
routeCases = readtable(fullfile(projectFolder,"targeted_reviewer_experiments", ...
    "nasa_qi_role_ablation","qi_role_cases.csv"),VariableNamingRule="preserve");
figure2Cases = routeCases(:,["caseList","caseSampleCount", ...
    "AllTimeMAE","RouteOnlyMAE"]);
figure2Cases.Properties.VariableNames = ["CaseID","SampleCount", ...
    "FixedAllSensorPlusTimeMAE_mm","SQTR_MAE_mm"];
figure2Cases.SQTRMinusComparator_MAE_mm = ...
    figure2Cases.SQTR_MAE_mm - figure2Cases.FixedAllSensorPlusTimeMAE_mm;
writetable(figure2Cases,fullfile(outputFolder,"figure2b_verification.csv"));

%% Randomised degradation and fixed-grid Oracle audit
randomStats = readtable(fullfile(projectFolder,"additional_validation_results", ...
    "reviewer_improvements","random_fault_statistics.csv"), ...
    VariableNamingRule="preserve",TextType="string");
randomCombined = randomStats(randomStats.Dataset == "Combined",:);
assert(height(randomCombined) == 1 && randomCombined.IndependentUnitN == 12);

fixedStats = readtable(fullfile(projectFolder,"multifault_robustness_results", ...
    "multifault_statistics.csv"),VariableNamingRule="preserve",TextType="string");
fixedCombined = fixedStats(fixedStats.Dataset == "Combined",:);
assert(height(fixedCombined) == 1 && fixedCombined.IndependentExperimentN == 12);

clean = readtable(fullfile(projectFolder,"multifault_robustness_results", ...
    "multifault_clean_gating_summary.csv"),VariableNamingRule="preserve", ...
    TextType="string");
cleanWide = unstack(clean,"MAE","Model");
cleanWide.MAEDifferenceAutoMinusFixed = cleanWide.AutoGatedClean - cleanWide.FullClean;
writetable(cleanWide,fullfile(outputFolder,"clean_false_trigger_mae_audit.csv"));

%% Training-fold alarm calibration checks
alarmFolder = fullfile(projectFolder,"targeted_reviewer_experiments", ...
    "alarm_calibration");
alarmSelections = readtable(fullfile(alarmFolder,"alert_threshold_selections.csv"), ...
    VariableNamingRule="preserve");
assert(height(alarmSelections) == 16*4*3);
assert(all(alarmSelections.InnerPositiveRecordN > 0));
assert(all(isfinite(alarmSelections.SelectedScoreThreshold)));
assert(all(alarmSelections.AchievedInnerSensitivity + 1e-12 >= ...
    alarmSelections.TargetInnerSensitivity));
alarmSummary = groupsummary(alarmSelections, ...
    ["WearThreshold","TargetInnerSensitivity"], ...
    ["min","max","mean"], ...
    ["SelectedScoreThreshold","InnerPositiveRecordN", ...
    "AchievedInnerSensitivity"]);
writetable(alarmSummary,fullfile(outputFolder,"alarm_threshold_selection_audit.csv"));

alarmStats = readtable(fullfile(alarmFolder,"alert_statistical_results.csv"), ...
    VariableNamingRule="preserve",TextType="string");

%% Compact machine-readable key-result ledger
result = strings(9,1);
value = strings(9,1);
source = strings(9,1);
result(1) = "NASA SQTR pooled MAE";
value(1) = sprintf("%.6f mm",sqtrPooled);
source(1) = "mill/results/time_model_overall.csv";
result(2) = "Input-matched fixed full-sensor plus-time pooled MAE";
value(2) = sprintf("%.6f mm",allTimePooled);
source(2) = "mill/results/time_model_overall.csv";
result(3) = "Input-matched pooled MAE reduction";
value(3) = sprintf("%.4f%%",routeReduction);
source(3) = "targeted_reviewer_experiments/nasa_qi_role_ablation/qi_role_route_comparisons.csv";
result(4) = "Input-matched case-macro difference (SQTR - comparator)";
value(4) = sprintf("%.6f mm; 95%% CI [%.6f, %.6f]; Holm p=%.6f", ...
    route.MacroMAEDifferenceBMinusA,route.MacroDifferenceCILower, ...
    route.MacroDifferenceCIUpper,route.HolmP);
source(4) = "targeted_reviewer_experiments/nasa_qi_role_ablation/qi_role_route_comparisons.csv";
result(5) = "No-time minus SQTR case-macro difference";
value(5) = sprintf("%.6f mm; 95%% CI [%.6f, %.6f]; exact p=%.8f", ...
    timeAudit.MeanCaseMacroDifferenceNoTimeMinusSQTR_mm, ...
    timeAudit.BootstrapCILower_mm,timeAudit.BootstrapCIUpper_mm,timeAudit.ExactWilcoxonP);
source(5) = "audit_outputs/time_ablation_case_macro_audit.csv";
result(6) = "Randomised-degradation relative-MAE reduction";
value(6) = sprintf("%.4f%%; Holm p=%.8f", ...
    randomCombined.RelativeImprovementPercent,randomCombined.HolmAdjustedP);
source(6) = "additional_validation_results/reviewer_improvements/random_fault_statistics.csv";
result(7) = "Fixed-grid relative MAE (fixed, automatic, Oracle)";
value(7) = sprintf("%.6f, %.6f, %.6f",fixedCombined.FullRelativeMAE, ...
    fixedCombined.AutoRelativeMAE,fixedCombined.OracleRelativeMAE);
source(7) = "multifault_robustness_results/multifault_statistics.csv";
result(8) = "Primary alert miss-rate difference";
alertMiss = alarmStats(alarmStats.Endpoint == "Alert miss rate",:);
value(8) = sprintf("%.6f; 95%% CI [%.6f, %.6f]; Holm p=%.6f", ...
    alertMiss.MeanDifferenceCalibratedMinusUncalibrated, ...
    alertMiss.BootstrapCILower,alertMiss.BootstrapCIUpper,alertMiss.HolmAdjustedP);
source(8) = "targeted_reviewer_experiments/alarm_calibration/alert_statistical_results.csv";
result(9) = "Primary false-positive-rate difference";
falsePositive = alarmStats(alarmStats.Endpoint == "False-positive rate",:);
value(9) = sprintf("%.6f; 95%% CI [%.6f, %.6f]; Holm p=%.6f", ...
    falsePositive.MeanDifferenceCalibratedMinusUncalibrated, ...
    falsePositive.BootstrapCILower,falsePositive.BootstrapCIUpper, ...
    falsePositive.HolmAdjustedP);
source(9) = "targeted_reviewer_experiments/alarm_calibration/alert_statistical_results.csv";
ledger = table(result,value,source,VariableNames=["Result","VerifiedValue","AuthoritativeSource"]);
writetable(ledger,fullfile(outputFolder,"key_result_ledger.csv"));

fprintf("V17 audit passed. Outputs: %s\n",outputFolder);
