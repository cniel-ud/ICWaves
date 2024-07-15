clearvars; eeglab; close;

expr = 'sub-(?<id>\d{2}).*';
data_dir = getenv('DATA_DIR');
out_dir = fullfile(data_dir, 'raw_data_and_IC_labels');
if ~isfolder(out_dir), mkdir(out_dir), end

file_list = dir(fullfile(data_dir, 'sub-*/eeg/*.set'));
n_subj = length(file_list);


for ii = 1:length(file_list)
    subj = regexp(file_list(ii).name, expr, 'names');
    subjID = subj.id;
    EEG = pop_loadset(...
            'filename', file_list(ii).name, ...
            'filepath', file_list(ii).folder, ...
            'check', 'off',...
            'verbose', 'off');

    srate = EEG.srate;

    % Data needed to compute ICs
    data = EEG.data;
    icaweights = EEG.icaweights;
    icasphere = EEG.icasphere;

     % Get ICLabel (noisy) labels
    EEG = iclabel(EEG);
    noisy_labels = EEG.etc.ic_classification.ICLabel.classifications;
    % Get winner class from ICLabel
    [~, winner_label] = max(noisy_labels, [], 2);

    n_ics = size(icaweights, 1);

    % Build mixed-source labels
    % First, add the expert labels:
    labels = -1*ones(n_ics, 1);
    labels(EEG.gdcomps) = 1; % Brain
    labels(EEG.muscle) = 2; % Muscle
    labels(EEG.blink) = 3; % Eye
    labels(EEG.lateyes) = 3; % Eye

    % use ICLabel where no expert label exist
    expert_label_mask = labels > 0;
    labels(~expert_label_mask) = winner_label(~expert_label_mask); % use ICLabel where no expert label exist


    out_file = fullfile(out_dir, sprintf('subj-%s.mat', subjID));
    fprintf('out_file = %s\n', out_file);
    save(out_file, 'data', 'srate', 'icaweights', 'icasphere', 'noisy_labels', 'labels', 'expert_label_mask', '-v7')
    fprintf('File %s created\n', out_file);
end