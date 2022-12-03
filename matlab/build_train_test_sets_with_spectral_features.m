clearvars; eeglab; close;
% Move to folder where this file is located
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end

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
test_subjects = 1:7;

expr = 'sub-(?<id>\d{2}).*';
data_dir = '../data/ds003004';
out_dir = fullfile(data_dir, 'spectral_features');
if ~isfolder(out_dir), mkdir(out_dir), end

file_list = dir(fullfile(data_dir, 'sub-*/eeg/*.set'));
n_subj = length(file_list);

X_train = [];
X_test = [];
noisy_labels_train = [];
noisy_labels_test = [];
y_train = [];
y_test = [];
expert_label_mask_train = [];
expert_label_mask_test = [];
subj_ind_ar_train = [];
subj_ind_ar_test = [];

for ii = 1:length(file_list)
    subj = regexp(file_list(ii).name, expr, 'names');
    subjID = subj.id;
    EEG = pop_loadset(...
            'filename', file_list(ii).name, ...
            'filepath', file_list(ii).folder, ...
            'check', 'off',...
            'verbose', 'off');
    
    % Get winner class from ICLabel
    noisy_tmp = EEG.etc.ic_classification.ICLabel.classifications;
    [~, winner_label] = max(noisy_tmp, [], 2);
    
    n_ics = size(EEG.icaact, 1);
    % Build mixed-source labels
    % First, add the expert labels:
    y_tmp = -1*ones(n_ics, 1);                
    y_tmp(EEG.gdcomps) = 1; % Brain
    y_tmp(EEG.muscle) = 2; % Muscle
    y_tmp(EEG.blink) = 3; % Eye
    y_tmp(EEG.lateyes) = 3; % Eye 
    % then, add ICLabel labels
    has_expert_label = y_tmp > 0;
    y_tmp(~has_expert_label) = winner_label(~has_expert_label); % use ICLabel where no expert label exist
    
    % Compute [PSD autocorrelation] feature vector
    features = extract_psd_and_autocorr(EEG, true);
    
    % Keep track of subject ID for each IC. This would be needed during
    % leave-one-subject-out cross-validation    
    subj_tmp = ones(n_ics, 1) * str2double(subjID);    
    
    if any(test_subjects == str2double(subjID))
        X_test = [X_test; features];        
        expert_label_mask_test = [expert_label_mask_test; has_expert_label];                
        y_test = [y_test; y_tmp];
        noisy_labels_test = [noisy_labels_test; noisy_tmp];
        subj_ind_ar_test = [subj_ind_ar_test; subj_tmp];
    else % train set.
        X_train = [X_train; features];        
        expert_label_mask_train = [expert_label_mask_train; has_expert_label];                        
        y_train = [y_train; y_tmp];
        noisy_labels_train = [noisy_labels_train; noisy_tmp];
        subj_ind_ar_train = [subj_ind_ar_train; subj_tmp];
    end
end
expert_label_mask_train = logical(expert_label_mask_train);
expert_label_mask_test = logical(expert_label_mask_test);
fpath = fullfile(out_dir, 'train_data.mat');
save(fpath, 'X_train', 'y_train', 'expert_label_mask_train', 'subj_ind_ar_train', 'noisy_labels_train', '-v7');
fpath = fullfile(out_dir, 'test_data.mat');
save(fpath, 'X_test', 'y_test', 'expert_label_mask_test', 'subj_ind_ar_test', 'noisy_labels_test', '-v7');