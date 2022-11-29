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
out_dir = fullfile(data_dir, 'icact_iclabel');
if ~isfolder(out_dir), mkdir(out_dir), end

file_list = dir(fullfile(data_dir, 'sub-*/eeg/*.set'));

shift = 0;
for ii = 1:length(file_list)
    subj = regexp(file_list(ii).name, expr, 'names');
    subjID = subj.id;
    EEG = pop_loadset(...
            'filename', file_list(ii).name, ...
            'filepath', file_list(ii).folder, ...
            'check', 'off',...
            'verbose', 'off');
    icaact = EEG.icaact;
    n_ics = EEG.nbchan;
    noisy_labels = EEG.etc.ic_classification.ICLabel.classifications;
    expert_labels = -1*ones(n_ics,1);
    expert_labels(EEG.gdcomps) = 1; % Brain
    expert_labels(EEG.muscle) = 2; % Muscle
    expert_labels(EEG.blink) = 3; % Eye
    expert_labels(EEG.lateyes) = 3; % Eye

    if any(test_subjects == str2double(subjID))
        fname = sprintf('test_subj-%s.mat', subjID);
    else % train set.
        fname = sprintf('train_subj-%s.mat', subjID);
    end

    fpath = fullfile(out_dir, fname);
    save(fpath, 'icaact', 'noisy_labels', 'expert_labels', '-v7');
end
