clearvars; eeglab; close;
expr = 'frolich_extract_(?<id>\d{2}).mat';
data_path = getenv('DATA_DIR');
cue_path = fullfile(data_path, "cue");
expert_labels_path = fullfile(cue_path, 'expert_labels');
EEGstruct_path = fullfile(cue_path, 'EEG_struct');
out_dir = fullfile(cue_path, 'BIDS');
file_list = dir(expert_labels_path);
file_list([1, 2]) = []; % removes '.' and '..'
n_files = length(file_list);
cue_expert_labels = {'blink', 'neural', 'heart', 'lateyes', 'muscle', 'mixed'};
for i = 1:n_files
    load(fullfile(expert_labels_path, file_list(i).name)); %load X, W, labels, and classes
    subj = regexp(file_list(i).name, expr, 'names');
    subjID = subj.id;
    fname = sprintf('frolich_EEGlab_%s.mat', subjID);
    EEGstruct_file = fullfile(EEGstruct_path, fname);
    load(EEGstruct_file);

    EEG.expert_ica_labels = labels;
    EEG.ica_classes = classes;

    out_file = sprintf('subj-%s.set', subjID);
    pop_saveset(EEG, 'filepath', char(out_dir), 'filename', out_file)
    fprintf('File %s created\n', out_file);
end
