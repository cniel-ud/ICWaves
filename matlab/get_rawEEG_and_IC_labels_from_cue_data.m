clearvars; eeglab; close;
expr = 'frolich_extract_(?<id>\d{2}).mat';
data_path = getenv('DATA_DIR');
cue_path = fullfile(data_path, "cue");
expert_labels_path = fullfile(cue_path, 'expert_labels')
EEGstruct_path = fullfile(cue_path, 'EEG_struct');
out_dir = fullfile(cue_path, 'raw_data_and_IC_labels');
file_list = dir(expert_labels_path)
n_files = length(file_list);
cue_expert_labels = {'blink', 'neural', 'heart', 'lateyes', 'muscle', 'mixed'};
for i = 1:n_files
  if ~(strcmp(file_list(i).name, '.') || strcmp(file_list(i).name, '..'))
    load(fullfile(expert_labels_path, file_list(i).name)); %load X, W, labels, and classes
    cue_labels = labels;
    subj = regexp(file_list(i).name, expr, 'names');
    subjID = subj.id;
    fname = sprintf('frolich_EEGlab_%s.mat', subjID);
    EEGstruct_file = fullfile(EEGstruct_path, fname);
    load(EEGstruct_file);

    % Data needed to compute ICs
    data = EEG.data;
    icaweights = EEG.icaweights;
    icasphere = EEG.icasphere;

    srate = EEG.srate;

    % Get ICLabel (noisy) labels
    EEG = iclabel(EEG);
    noisy_labels = EEG.etc.ic_classification.ICLabel.classifications;
    % Get winner class from ICLabel
    [~, winner_label] = max(noisy_labels, [], 2);
    
    n_ics = size(EEG.icaweights, 1);
    % Build mixed-source labels
    labels = -1*ones(n_ics, 1);
    % First, add the expert labels.
    % Map `cue` classes to `emotion_study` classes
    labels(cue_labels == 2) = 1; % Brain
    labels(cue_labels == 5) = 2; % Muscle
    labels(cue_labels == 1) = 3; % Eye
    labels(cue_labels == 4) = 3; % Eye
    labels(cue_labels == 3) = 4; % Heart 
    % then, add ICLabel labels
    % The `mixed` class in `cue` is populated with noisy labels             
    expert_label_mask = labels > 0;
    labels(~expert_label_mask) = winner_label(~expert_label_mask); % use ICLabel where no expert label exist
    
    out_file = fullfile(out_dir, sprintf('subj-%s.mat', subjID));
    fprintf('out_file = %s\n', out_file);
    save(out_file, 'data', 'srate', 'icaweights', 'icasphere', 'noisy_labels', 'labels', 'expert_label_mask', '-v7')
    fprintf('File %s created\n', out_file);
  end
end
