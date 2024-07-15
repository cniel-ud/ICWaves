clearvars; eeglab; close;

% test segment length in seconds
test_segment_len = [10,   20,   30,   40,   50,   60,  ...
                    120,  180,  240,  300,  600,  900, ...
                    1200, 1500, 1800, 2100, 2400, 2700, ...
                    3000];

expr = 'sub-(?<id>\d{2}).*';
data_dir = getenv('DATA_DIR');

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

    % Build mixed-source labels
    % First, add the expert labels:
    n_ics = size(EEG.icaweights, 1);
    labels = -1*ones(n_ics, 1);
    labels(EEG.gdcomps) = 1; % Brain
    labels(EEG.muscle) = 2; % Muscle
    labels(EEG.blink) = 3; % Eye
    labels(EEG.lateyes) = 3; % Eye

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


        dir_name = sprintf("IC_labels_at_%d_seconds", test_segment_len(jj));
        out_dir = fullfile(data_dir, dir_name);
        if ~isfolder(out_dir), mkdir(out_dir), end
        out_file = fullfile(out_dir, sprintf('subj-%s.mat', subjID));
        fprintf('out_file = %s\n', out_file);
        save(out_file, 'noisy_labels', 'labels', 'expert_label_mask', '-v7')
        fprintf('File %s created\n', out_file);
    end
end