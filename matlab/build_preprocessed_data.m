clearvars; eeglab; close;

ICLabels = {...
    'Brain', 'Muscle', 'Eye', 'Heart',...
    'Line Noise', 'Channel Noise', 'Other'};

% Subjects s-{01,02,..,07} have in total 1469 ICs, and 227 of those have
% expert labels, with ratios
% brain   muscle  eye
% 0.5639  0.3744  0.0617
% which is more or less the same respect to the whole set of expert
% annotated ICs:
% brain   muscle  eye
% 0.6007  0.3365  0.0628
% Here eye = blink + lateyes
% 227 is 19.54% of the whole set of expert annotated ICs (1162)
all_subjects_subjects = 1:7;

expr = 'sub-(?<id>\d{2}).*';
data_dir = getenv('DATA_DIR');
out_dir = fullfile(data_dir, 'preprocessed_data');
if ~isfolder(out_dir), mkdir(out_dir), end

file_list = dir(fullfile(data_dir, 'sub-*/eeg/*.set'));
n_subj = length(file_list);

icaact_all_subjects = [];
y_all_subjects = [];
psd_and_autocorr_all_subjects = [];
ICLabels_all_subjects = [];
mixed_labels_all_subjects = [];
expert_label_mask_all_subjects = [];
subj_ind_all_subjects = [];
pnts_all_subjects = zeros(1, n_subj);
srate_all_subjects = zeros(1, n_subj);
n_ics_all_subjects = zeros(1, n_subj);

for ii = 1:length(file_list)
    subj = regexp(file_list(ii).name, expr, 'names');
    subjID = subj.id;
    EEG = pop_loadset(...
            'filename', file_list(ii).name, ...
            'filepath', file_list(ii).folder, ...
            'check', 'off',...
            'verbose', 'off');

    srate_all_subjects(ii) = EEG.srate;
    pnts_all_subjects(ii) = EEG.pnts;

    % Get winner class from ICLabel
    ICLabel_scores = EEG.etc.ic_classification.ICLabel.classifications;
    [~, winner_ICLabel_label] = max(ICLabel_scores, [], 2);

    icaact = EEG.icaact;
    icaact_all_subjects = [icaact_all_subjects; icaact];
    n_ics = size(icaact, 1);
    n_ics_all_subjects(ii) = n_ics;

    % Build mixed-source labels
    % First, add the expert labels:
    mixed_labels = -1*ones(n_ics, 1);
    mixed_labels(EEG.gdcomps) = 1; % Brain
    mixed_labels(EEG.muscle) = 2; % Muscle
    mixed_labels(EEG.blink) = 3; % Eye
    mixed_labels(EEG.lateyes) = 3; % Eye

    % then, add ICLabel labels
    has_expert_label = mixed_labels > 0;
    % use ICLabel where no expert label exist
    mixed_labels(~has_expert_label) = winner_ICLabel_label(~has_expert_label);

    % Compute [PSD autocorrelation] feature vector
    psd_and_autocorr = extract_psd_and_autocorr(EEG, true);

    % Keep track of subject ID for each IC. This would be needed during
    % leave-one-subject-out cross-validation
    subj_ind = ones(n_ics, 1) * str2double(subjID);

    psd_and_autocorr_all_subjects = [psd_and_autocorr_all_subjects; psd_and_autocorr];
    expert_label_mask_all_subjects = [expert_label_mask_all_subjects; has_expert_label];
    mixed_labels_all_subjects = [mixed_labels_all_subjects; mixed_labels];
    ICLabels_all_subjects = [ICLabels_all_subjects; winner_ICLabel_label];
    subj_ind_all_subjects = [subj_ind_all_subjects; subj_ind];
end
expert_label_mask_all_subjects = logical(expert_label_mask_all_subjects);
fpath = fullfile(out_dir, 'data.mat');
save(fpath, 'icaact_all_subjects', 'psd_and_autocorr_all_subjects', ...
            'mixed_labels_all_subjects', 'expert_label_mask_all_subjects', ...
            'subj_ind_all_subjects', 'ICLabels_all_subjects', ...
            'srate_all_subjects', 'pnts_all_subjects', 'n_ics_all_subjects', '-v7');