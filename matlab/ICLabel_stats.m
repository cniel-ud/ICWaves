% Get ICLabel statistics from all subjects:
% * Number of ICs per class
% * Min, Max, and mean legth of ICs
clearvars; eeglab; close;
% Move to folder where this file is located
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end

data_dir = '../data/ds003004';
file_list = dir(fullfile(data_dir, 'sub-*/eeg/*.set'));

ICLabels = {...
    'Brain', 'Muscle', 'Eye', 'Heart',...
    'Line Noise', 'Channel Noise', 'Other'};

expert_label_fields = {'gdcomps', 'muscle', 'blink', 'lateyes'};

prob_thr = [0.5, 0.6, 0.7, 0.8, 0.9];
class_count = zeros(7, length(prob_thr));
ic_length = zeros(length(file_list), 1);
ic_count = zeros(length(file_list), 1);
sampling_rates = zeros(length(file_list), 1);
expert_labels_count = zeros(length(file_list), 4);
winner_noisy_labes = cell(length(file_list),1);
ICLabel_errors = cell(length(file_list), 3);

for ii = 1:length(file_list)
    EEG = pop_loadset(...
        'filename', file_list(ii).name, ...
        'filepath', file_list(ii).folder, ...
        'loadmode', 'info',...
        'check', 'off',...
        'verbose', 'off');
    for jj = 1:length(prob_thr)
        [r, c] = find(EEG.etc.ic_classification.ICLabel.classifications > prob_thr(jj));
        [C,ia,ic] = unique(c);
        a_counts = accumarray(ic,1);
        class_count(C, jj) = class_count(C, jj) + a_counts;
    end
    ic_count(ii) = EEG.nbchan;
    ic_length(ii) = EEG.pnts;
    noisy_labels = EEG.etc.ic_classification.ICLabel.classifications;
    [~, winner_tmp] = max(noisy_labels, [], 2);
    winner_noisy_labes{ii} = winner_tmp;
    labeled_as_brain = winner_tmp == 1;
    labeled_as_muscle = winner_tmp == 2;
    labeled_as_eye = winner_tmp == 3;
    labeled_as_brain(EEG.gdcomps) = false; % ignore expert-annotated labels
    labeled_as_muscle(EEG.muscle) = false;
    labeled_as_eye(EEG.lateyes) = false;
    labeled_as_eye(EEG.blink) = false;
    ICLabel_errors{ii, 1} = sum(labeled_as_brain);
    ICLabel_errors{ii, 2} = sum(labeled_as_muscle);
    ICLabel_errors{ii, 3} = sum(labeled_as_eye);
    sampling_rates(ii) = EEG.srate;
    for jj =1:length(expert_label_fields)
        n_labels = length(EEG.(expert_label_fields{jj}));
        expert_labels_count(ii, jj) = n_labels;
    end
end

var_names = arrayfun(@(x) num2str(x), prob_thr,'UniformOutput', false);
row_names = [ICLabels, 'Total'];
tab_data = [class_count; sum(class_count, 1)];
components_per_class = array2table(tab_data, 'RowNames', row_names, 'VariableNames', var_names);
disp(components_per_class);

tab_data = [ic_count, ic_length, sampling_rates];
% row_names = file_list;
var_names = {'Count', 'Length', 'Fs'};
components_size = array2table(tab_data, 'VariableNames', var_names);
disp(components_size);

var_names = {'brain', 'muscle', 'blink', 'lateyes'};
expert_labels = array2table(expert_labels_count, 'VariableNames', var_names);
disp(expert_labels);

expert_labels_count_eyelumped = [...
    expert_labels_count(:,1:2),...
    sum(expert_labels_count(:,3:4), 2)
];
expert_labels_totals = sum(expert_labels_count_eyelumped, 2);