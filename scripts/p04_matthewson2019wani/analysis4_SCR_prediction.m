%% PCR
% The purpose of this script is to benchmark Wani's PCR SCR analysis. 
% We already have the data preprocessed in the sense that:
% 1) the data is organized per onset
% 2) we downsampled the data
% Original code can be accessed here: https://github.com/canlab/cognitive_regulation_physiology

% THOUGHTS: --> TODO: organize with heejung
% signal_m: average trial per participant, factor 1 and factor 2. 


%--------------------------------------------------------------------------
%                             toolboxes
%--------------------------------------------------------------------------

% necessary toolboxes:
% 1. Wani Repo: https://github.com/wanirepo/wagerlabtools_supplement
% 2. CanLabCore: https://github.com/canlab/CanlabCore
% 3. Mediation toolbox: https://github.com/canlab/MediationToolbox 

%%
%---------------------------------------------------------------------------
%                       load & organize data 
%--------------------------------------------------------------------------

clear all;
close all;

% ===================== define directories ================================


% ========================= load data =====================================

data_scl = readtable('/Users/isabelneumann/Documents/PhD/HEEJUNG/sub-all_ses-all_run-all_runtype-pain_epochstart--3_epochend-20_samplingrate-25_ttlindex-1_physio-scltimecourse.csv');
data_qc = readtable('/Users/isabelneumann/Documents/PhD/HEEJUNG/QC_EDA_new.csv'); 

% =============== use only participants with good quality =================

subset = data_qc(~strcmp(data_qc.SignalQuality, 'exclude'), :);

data = innerjoin(data_scl, subset, 'Keys', {'src_subject_id', 'session_id', 'param_task_name', 'param_run_num'});
rowsToRemove = isnan(data.event04_actual_angle);
data(rowsToRemove, :) = [];

% ====================== define parameters ================================

sampling_rate = 25;
epoch_start = -3;
epoch_end = 20;
baseline_timepoint = abs(epoch_start)*sampling_rate;
total_timepoint = abs(epoch_end-epoch_start)*sampling_rate;

% TODO: add a column to exclude participants without behavioral data  

% Find rows with the specific value in a column
% matchingRows = data_qc(strcmp(data_qc.SignalQuality, 'exclude'), :);
% Merge data with qc
% [~, idx1, idx2] = intersect(matchingRows(:, {'src_subject_id', 'session_id', 'param_task_name', 'param_run_num'}), ...
    %data(:, {'src_subject_id', 'session_id', 'param_task_name', 'param_run_num'}), 'rows');

% rowsToExclude = ismember([data.src_subject_id, data.session_id, data.param_task_name, data.param_run_num], ...
%     [matchingRows.src_subject_id, matchingRows.session_id, matchingRows.param_task_name, matchingRows.param_run_num], 'rows')
% data_without_QC = data(~rowsToExclude, :);
% 
% qcdata = data;
% qcdata(idx1, :) = []; -> TODO: Heejung: this section be deleted?

% ====================== define conditions ================================

data.temp(categorical(data.param_stimulus_type) == 'low_stim') = 48;
data.temp(categorical(data.param_stimulus_type) == 'high_stim') = 49;
data.temp(categorical(data.param_stimulus_type) == 'med_stim') = 50;

data.cond(categorical(data.param_task_name) == 'pain') = 1;
data.cue(categorical(data.param_cue_type) == 'high_cue') = 1;
data.cue(categorical(data.param_cue_type) == 'low_cue') = -1;

% ================ define subjects with full data set =====================

sub_list = unique(data.src_subject_id);
sub_list_clean = sub_list(~isnan(sub_list));

for i = 1:numel(sub_list_clean)
    y_int{i} = data.event04_actual_angle(data.src_subject_id == sub_list_clean(i)); % intensity rating
    y_exp{i} = data.event02_expect_angle(data.src_subject_id == sub_list_clean(i)); % expectation rating
    cue{i} = data.cue(data.src_subject_id == sub_list_clean(i)); %RENAME: cue for reg
    temp{i} = data.temp(data.src_subject_id == sub_list_clean(i));
end

%%
%--------------------------------------------------------------------------
%                  calculate subject mean per condition
%--------------------------------------------------------------------------

u_temp = unique(temp{1});
u_cue = unique(cue{1}); 
dat_int = {};
dat_exp = {};
for subj = 1:numel(temp)
    for i = 1:numel(u_temp)
        for j = 1:numel(u_reg)
            dat_int{i,j}(subj,1) = nanmean(y_int{subj}(temp{subj}==u_temp(i) & cue{subj}==u_cue(j)));
            dat_exp{i,j}(subj,1) = nanmean(y_exp{subj}(temp{subj}==u_temp(i) & cue{subj}==u_cue(j)));
        end
    end
end

%%
%--------------------------------------------------------------------------
%                  creating x: SCR data (averaged over three trials) 
%--------------------------------------------------------------------------

% ===================== get the column indices ============================
startColumn = 'time_0';
endColumn = 'time_574';  

startIndex = find(strcmp(data.Properties.VariableNames, startColumn));
endIndex = find(strcmp(data.Properties.VariableNames, endColumn));

temp_list = {'low_stim', 'med_stim', 'high_stim'};
cue_list = {'low_cue', 'high_cue'};

% ==================== specify number of subjects =========================

numSubjects = length(sub_list_clean); 
signal = cell(1, numSubjects);

% ======================== loop over subjects =============================

for i = 1:numSubjects % initialize the cond structure for each subject
    signal{i}.cond = cell(3, 2); 
end


for j =  1:numel(sub_list_clean)
    for t  = 1:length(temp_list)
        for c = 1:length(cue_list)
      
            signal{1,j}.cond{t, c}(:,:) =  table2array(data(data.src_subject_id == sub_list_clean(j) &  strcmp(data.param_cue_type,cue_list(c)) &  strcmp(data.param_stimulus_type,temp_list(t)), startIndex:endIndex));
            % get data for 23 seconds
        end
    end
end

% ======================== baseline correction=============================
% Snipping out 23 seconds in total. 0 is the onset of occurrence; the first
% 3 seconds (sec -3 to sec 0) are used for baseline (= calculate the mean within subject per 
% condition and subtract) 

clear signal_m;
k = 0;

for i = 1:length(temp_list)     % temperature: 48, 49, 50
    for j = 1:length(cue_list)  % cue condition: low cue, high cue
        k = k + 1;
        for subj = 1:numel(signal)
            signal_m{i,j}(subj,:) = nanmean(signal{subj}.cond{i,j} - repmat(mean(signal{subj}.cond{i,j}(:,1:baseline_timepoint),2), 1, size(signal{subj}.cond{i,j},2)));
            % subtracting the baseline, and then average
        end
    end
end

%%
%--------------------------------------------------------------------------
%                        step 3: running PCR
%--------------------------------------------------------------------------

% ========= 3.1: determining the number of components of the model ========

addpath(genpath('/Users/isabelneumann/Dokumente/CanlabCore/CanlabCore'));


for i = 1:length(temp_list)
    for j = 1:length(cue_list)
        subjs{i,j} = (1:size(dat_int{i,j},1))';  % ready for leave-one-participant-out (LOPO) cross-validation
    end
end

% NOTE: in our case, dat_int{:,1} is low cue and dat_int{:,2} is high cue

dat.int = fmri_data; % to use fmri_data.predict function, 
                     % put the data in fmri_data object
dat.int.Y = cat(1,dat_int{:,1});    % add concatenated intensity ratings in dat.Y
dat.int.dat = cat(1,signal_m{:,1})'; % add concatenated epoch data in dat.dat: [ 25*20sec SCR X number_participants* 3 condition ]

% NOTE: For the training, we only used passive experience runs.

dat.int.dat = dat.int.dat(baseline_timepoint+1:total_timepoint,:); % Stimulus-locked 20 seconds epoch 
whfolds = cat(1,subjs{:,1});  %cat(1,subjs{:,2});                  % cross-validaion folds (LOPO CV) [ rep(41sub x 6conditions) ]

dat.exp = dat.int;
dat.exp.Y = cat(1,dat_exp{:,1});%cat(1,dat_exp{:,2});  % separate code & predict rating from medium intensity , second column of temp
datdir = '/Users/isabelneumann/Dokumente';

% savename = fullfile(datdir, 'SCR_prediction_dat_112816.mat');
% save(savename, '-append', 'dat', 'whfolds', 'dat_int', 'dat_exp', 'signal_m');
% 
% clear rmse pred_outcome_r;

for i = 2:10
    if i == 2, disp('SCR predictive model for intensity ratings'); end
    [~, stats_int, ~] = predict(dat.int, 'algorithm_name', 'cv_pcr', 'nfolds', whfolds, 'numcomponents', i, 'verbose', 0);
    rmse.int(i-1) = stats_int.rmse;
    
    for j = unique(whfolds)'
        por_subj(j) = corr(stats_int.Y(whfolds==j), stats_int.yfit(whfolds==j));
    end
    
    pred_outcome_r.int(i-1) = mean(por_subj);
    fprintf('\n #component = %d, rmse = %1.4f, pred_outcome_r = %1.4f', i, stats_int.rmse, mean(por_subj));
end
clear por_subj;

for i = 2:10
    if i == 2, disp('SCR predictive model for expectation ratings'); end
    [~, stats_exp, ~] = predict(dat.exp, 'algorithm_name', 'cv_pcr', 'nfolds', whfolds, 'numcomponents', i, 'verbose', 0);
    rmse.exp(i-1) = stats_exp.rmse;
    
    for j = unique(whfolds)'
        por_subj(j) = corr(stats_exp.Y(whfolds==j), stats_exp.yfit(whfolds==j));
    end
    
    pred_outcome_r.exp(i-1) = mean(por_subj);
    fprintf('\n #component = %d, rmse = %1.4f, pred_outcome_r = %1.4f', i, stats_exp.rmse, mean(por_subj));
end

%%
% === 3.2: plot for rmse and pred_outcome_r for different numbers of components ===

savefig = 0;

[~, x] = max(-scale(rmse.int)+scale(pred_outcome_r.int));

create_figure('por'); 

plot(2:10, pred_outcome_r.int, 'color', 'k', 'linewidth', 2);
if ~savefig
    xlabel('The number of principal components', 'fontsize', 20);
    ylabel('Prediction-outcome correlation', 'fontsize', 20);
end

y = pred_outcome_r.int(x);

hold on;
line([x+1 x+1], [0.35 y], 'linestyle', '--', 'color', [.4 .4 .4], 'linewidth', 1.5);
scatter(x+1, y, 150, 'r', 'marker', 's', 'markerfacecolor', 'r', 'markeredgecolor', 'k');
set(gca, 'tickDir', 'out', 'tickLength', [.02 .02], 'linewidth', 2, 'ylim', [-.840 .84], 'xlim', [1.5, 10.5], 'fontsize', 20);
if ~savefig
    set(gcf, 'position', [360   318   516   380]);
else
    set(gcf, 'position', [360   455   338   243]);
end

if savefig
    savename = fullfile(figdir, 'prediction_outcome_int_ncomp.pdf');
    pagesetup(gcf);
    saveas(gcf, savename);
    
    pagesetup(gcf);
    saveas(gcf, savename);
end

%% 
% ====================== 3.2: similar results for RMSE ======================

% close all;
create_figure('rmse'); 
plot(2:10, rmse.int, 'color', 'k', 'linewidth', 2);
xlabel('The number of principal components', 'fontsize', 22);
ylabel('RMSE', 'fontsize', 22);

hold on;

y = rmse.int(x);

line([x+1 x+1], [10.8 y], 'linestyle', '--', 'color', [.4 .4 .4], 'linewidth', 1.5);
scatter(x+1, y, 80, 'r', 'marker', 's', 'markerfacecolor', 'r', 'markeredgecolor', 'k');

set(gca, 'tickDir', 'out', 'tickLength', [.02 .02], 'linewidth', 2, 'fontsize', 20, 'xlim', [1.5 10.5]);

figdir = '/Users/isabelneumann/Documents/PhD/HEEJUNG';
% savename = fullfile(figdir, 'rmse_int_ncomp.pdf');
% try
%     pagesetup(gcf);
%     saveas(gcf, savename);
% catch
%     pagesetup(gcf);
%     saveas(gcf, savename);
% end

%%
% === 3.3: PCR model for intensity ratings with NCOMP = 6, bootstrapping 10000 ===

[~, pcr_stats.int, ~] = predict(dat.int, 'algorithm_name', 'cv_pcr', 'nfolds', whfolds, 'numcomponents', 6, 'bootweights', 'bootsamples', 10000); % run it again
[~, pcr_stats.exp, ~] = predict(dat.exp, 'algorithm_name', 'cv_pcr', 'nfolds', whfolds, 'numcomponents', 6, 'bootweights', 'bootsamples', 10000); % run it again

save(fullfile(datdir, 'SCR_prediction_dat_112816.mat'), '-append', 'pcr_stats', 'pred_outcome_r', 'rmse');


%%
% ========================= 3.3: plot for intensity =======================

% close all;
% 
% load(fullfile(datdir, 'SCR_prediction_dat_112816.mat'));

addpath(genpath('/Users/isabelneumann/Dokumente/wanirepo/wagerlabtools_supplement'));

x = 1:total_timepoint; % 1:500
create_figure('PCR weights');

for i = (5:5:20)*25 %TODO: may have to change index
    line([i i], [-.3 1], 'col', [.8 .8 .8], 'linewidth', 1);
end

sig_idx = pcr_stats.int.WTS.wP<getFDR(pcr_stats.int.WTS.wP, .05); % significant time points = xx [67:215 282:500]
idx{1} = 1:67;
idx{2} = 216:282;
idx{3} = 67:216;
idx{4} = 282:500;

% NOTE: not sure what other_output is and how it differs from
% other_output_cv. 
% I do see the description differnce, but don't understand the conceptual
% difference as to how this was obtained: "['Other output from algorithm -
% trained on all data (these depend on algorithm)']"

for i = 1:2, plot(x(idx{i}), pcr_stats.int.other_output{1}(idx{i}), 'color', [.3 .3 .3], 'linewidth', 3); end
for i = 3:4, plot(x(idx{i}), pcr_stats.int.other_output{1}(idx{i}), 'color', [0.8431    0.1882    0.1216], 'linewidth', 3); end

set(gcf, 'position', [50   208   423   242]);
set(gca, 'ylim', [-1 1], 'xlim', [0 500], 'linewidth', 1.5, 'TickDir', 'out', 'TickLength', [.02 .02], 'Xtick', (0:5:24)*sampling_rate, 'ytick', -1:.1:1); % ytick -.05:.05:.12
set(gca, 'XTickLabel', get(gca, 'XTick')./25);
set(gca, 'fontSize', 22);

savename = fullfile(figdir, 'SCR_intensity_predictive_weights.pdf');

pagesetup(gcf);
saveas(gcf, savename);

pagesetup(gcf);
saveas(gcf, savename);

%%
% ======================== 3.4: plot for expectation ======================
%%
% close all;

x = 1:500;
create_figure('PCR weights');

for i = (5:5:20)*25
    line([i i], [-.3 1], 'col', [.8 .8 .8], 'linewidth', 1);
end

sig_idx = pcr_stats.exp.WTS.wP<getFDR(pcr_stats.exp.WTS.wP, .05);
% significant time points = [278:500]
idx{1} = 1:278;
idx{2} = 278:375;
i=1; plot(x(idx{i}), pcr_stats.exp.other_output{1}(idx{i}), 'color', [.3 .3 .3], 'linewidth', 3); 
i=2; plot(x(idx{i}), pcr_stats.exp.other_output{1}(idx{i}), 'color', [0.8431    0.1882    0.1216], 'linewidth', 3); 

set(gcf, 'position', [50   208   423   242]);
set(gca, 'ylim', [-.08 .12], 'xlim', [0 500], 'linewidth', 1.5, 'TickDir', 'out', 'TickLength', [.02 .02], 'Xtick', (0:5:24)*25, 'ytick', -.05:.05:.12);
set(gca, 'XTickLabel', get(gca, 'XTick')./25);
set(gca, 'fontSize', 22);

savename = fullfile(figdir, 'SCR_expectation_predictive_weights.pdf');

pagesetup(gcf);
% saveas(gcf, savename);

pagesetup(gcf);
% saveas(gcf, savename);

fprintf('\ncorrelation between intensity and expectation weights r = %1.3f\n', corr(pcr_stats.int.other_output{1}, pcr_stats.exp.other_output{1}));


%%
%--------------------------------------------------------------------------
%                step 4: testing on (?) trials
%--------------------------------------------------------------------------

scr  = struct();
scr.signal_m = signal_m;
scr.pcr_stats = pcr_stats;
scr.dat_exp = dat_exp;
scr.dat_int = dat_int;


% Applying the model on regulation trials using leave-one-participant-out cross validation
for i = [1 3]
    for j = 1:6
        for subj = 1:size(scr.signal_m{j,i},1)
            test_scr_int{j,i}(subj,1) = scr.signal_m{j,i}(subj,26:375)*scr.pcr_stats.int.other_output_cv{subj,1} + scr.pcr_stats.int.other_output_cv{subj,2};
            test_scr_exp{j,i}(subj,1) = scr.signal_m{j,i}(subj,26:375)*scr.pcr_stats.exp.other_output_cv{subj,1} + scr.pcr_stats.exp.other_output_cv{subj,2};
        end
    end
end

temp_int = reshape(scr.pcr_stats.int.yfit, 41, 6);
temp_exp = reshape(scr.pcr_stats.exp.yfit, 41, 6);

for j = 1:6
    test_scr_int{j,2} = temp_int(:,j);
    test_scr_exp{j,2} = temp_exp(:,j);
end

% save(fullfile(datdir, 'SCR_prediction_dat_112816.mat'), '-append', 'test_scr_*');

% scatter plot
y_int = [cat(2,scr.dat_int{:,1}) cat(2,scr.dat_int{:,3})];
yfit_int = [cat(2,test_scr_int{:,1}) cat(2,test_scr_int{:,3})];

y_exp = [cat(2,scr.dat_exp{:,1}) cat(2,scr.dat_exp{:,3})];
yfit_exp = [cat(2,test_scr_int{:,1}) cat(2,test_scr_int{:,3})];

%%
colors = [255,237,160
254,217,118
254,178,76
253,141,60
252,78,42
227,26,28
189,0,38]./255;

xlim = [-2 65];
create_figure('predicted');

clear test_por;
for i = 1:size(y_int,1)
    hold on;
    x = y_int(i,:);
    y = yfit_int(i,:);
    b = glmfit(x,y);
    test_por(i) = corr(x',y');
end

dif = 1/size(colors,1);

k = zeros(size(test_por));
for i = 1:size(colors,1)
    idx = test_por <= (dif*i+.0001) & test_por >= dif*(i-1);
    k(idx) = i;
end

marker_shapes = repmat('osd^v><', 1, 40);

for i = 1:size(y_int,1)
    hold on;
    x = y_int(i,:);
    y = yfit_int(i,:);
    b = glmfit(x,y);
    line_h(i) = line(xlim, b'*[ones(1,2); xlim], 'linewidth', 1.5, 'color', colors(k(i),:)); % cmap(round(i*1.5),:));
    scatter(x, y, 120, colors(k(i),:), 'filled', 'markerfacealpha', .8, 'marker', marker_shapes(i));
end

line(xlim, xlim, 'linewidth', 4, 'linestyle', ':', 'color', [.5 .5 .5]);

set(gcf, 'position', [360   349   371   349]);
set(gca, 'tickdir', 'out', 'TickLength', [.02 .02], 'linewidth', 1.5, 'xlim', xlim, 'ylim', [-2 100], 'fontsize', 22);

savename = fullfile(figdir, 'SCR_actual_predicted_outcomes.pdf');

pagesetup(gcf);
saveas(gcf, savename);

pagesetup(gcf);
saveas(gcf, savename);


%% 
create_figure('predicted_exp');
dif = 1/size(colors,1);

clear test_por_exp;
for i = 1:size(y_exp,1)
    hold on;
    x = y_exp(i,:);
    y = yfit_exp(i,:);
    b = glmfit(x,y);
    test_por_exp(i) = corr(x',y');
end

k = zeros(size(test_por_exp));
for i = 1:size(colors,1)
    idx = test_por_exp <= (dif*i+.0001) & test_por_exp >= dif*(i-1);
    k(idx) = i;
end

marker_shapes = repmat('osd^v><', 1, 40);

for i = 1:size(y_exp,1)
    hold on;
    x = y_exp(i,:);
    y = yfit_exp(i,:);
    b = glmfit(x,y);
    try
        line_h(i) = line(xlim, b'*[ones(1,2); xlim], 'linewidth', 1.5, 'color', colors(k(i),:)); % cmap(round(i*1.5),:));
        scatter(x, y, 120, colors(k(i),:), 'filled', 'markerfacealpha', .8, 'marker', marker_shapes(i));
    catch
        line_h(i) = line(xlim, b'*[ones(1,2); xlim], 'linewidth', 1.5, 'color', [0.1961    0.5333    0.7412]); % cmap(round(i*1.5),:));
        scatter(x, y, 120, [0.1961    0.5333    0.7412], 'filled', 'markerfacealpha', .8, 'marker', marker_shapes(i));
    end
    
end

line(xlim, xlim, 'linewidth', 4, 'linestyle', ':', 'color', [.5 .5 .5]);

set(gcf, 'position', [360   349   371   349]);
set(gca, 'tickdir', 'out', 'TickLength', [.02 .02], 'linewidth', 1.5, 'xlim', xlim, 'ylim', [-2 100], 'fontsize', 22);

savename = fullfile(figdir, 'SCR_actual_predicted_outcomes_expectation.pdf');

pagesetup(gcf);
saveas(gcf, savename);

pagesetup(gcf);
saveas(gcf, savename);


fprintf('\nTest results: mean prediction_outcome_r for intensity = %1.3f', mean(test_por));
fprintf('\nTest results: mean prediction_outcome_r for expectation = %1.3f', mean(test_por_exp));
