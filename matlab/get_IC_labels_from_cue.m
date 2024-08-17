clearvars; eeglab; close;

% test segment length in seconds
test_segment_len =   [  9.,   19.5,   30.,   39.,   49.5,   60.,  120.,  ...
                      180.,   240.,  300.,  600.,   900., 1200., 1500., 1800., 2100.,...
                      2400., 2700. , 3000.];

expr = 'frolich_extract_(?<id>\d{2}).mat';
data_path = getenv('DATA_DIR');
cue_path = fullfile(data_path, "cue");
expert_labels_path = fullfile(cue_path, 'expert_labels');
EEGstruct_path = fullfile(cue_path, 'EEG_struct');
out_dir = fullfile(cue_path, 'raw_data_and_IC_labels');
file_list = dir(expert_labels_path);
file_list([1, 2]) = []; % removes '.' and '..'
n_files = length(file_list);
cue_expert_labels = {'blink', 'neural', 'heart', 'lateyes', 'muscle', 'mixed'};

n_subj = length(file_list);

for ii = 1:n_subj
    load(fullfile(expert_labels_path, file_list(ii).name)); %load X, W, labels, and classes
    cue_labels = labels;
    subj = regexp(file_list(ii).name, expr, 'names');
    subjID = subj.id;
    fname = sprintf('frolich_EEGlab_%s.mat', subjID);
    EEGstruct_file = fullfile(EEGstruct_path, fname);
    load(EEGstruct_file);
    srate = EEG.srate;

    % Build mixed-source labels
    n_ics = size(EEG.icaweights, 1);
    labels = -1*ones(n_ics, 1);
    % First, add the expert labels.
    % Map `cue` classes to `emotion_study` classes
    labels(cue_labels == 2) = 1; % Brain
    labels(cue_labels == 5) = 2; % Muscle
    labels(cue_labels == 1) = 3; % Eye
    labels(cue_labels == 4) = 3; % Eye
    labels(cue_labels == 3) = 4; % Heart

    % use ICLabel where no expert label exist
    expert_label_mask = labels > 0;

    EEG_tmp = EEG;

    for jj = 1:length(test_segment_len)
        % For a test segment length greater than 5 seconds, the computation
        % of the PSD and autocorr uses EEG.pnts to index the ICs.
        % In ICL_feature_extractor.m (a function in the ICLabel plugin), we
        % only need to add the following lines just before the PSD computation:
        %   if isfield(EEG, 'new_pnts')
        %       EEG.pnts = EEG.new_pnts;
        %   end
        new_pnts = floor(test_segment_len(jj) * srate);
        EEG_tmp.new_pnts = new_pnts;
        if new_pnts / EEG.srate <= 5
            error("Not Implmented: EEG.pnts / EEG.srate <= 5")
        end
        % Get ICLabel (noisy) labels
        EEG_tmp = iclabel(EEG_tmp);
        noisy_labels = EEG_tmp.etc.ic_classification.ICLabel.classifications;
        % Get winner class from ICLabel
        [~, winner_label] = max(noisy_labels, [], 2);

        labels(~expert_label_mask) = winner_label(~expert_label_mask); % use ICLabel where no expert label exist


        dir_name = sprintf("IC_labels_at_%.1f_seconds", test_segment_len(jj));
        out_dir = fullfile(cue_path, dir_name);
        if ~isfolder(out_dir), mkdir(out_dir), end
        out_file = fullfile(out_dir, sprintf('subj-%s.mat', subjID));
        fprintf('out_file = %s\n', out_file);
        save(out_file, 'noisy_labels', 'labels', 'expert_label_mask', '-v7')
        fprintf('File %s created\n', out_file);
    end
end