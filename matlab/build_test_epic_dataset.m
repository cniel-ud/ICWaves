home = getenv('HOME');
epic_path = fullfile(home, "personal_repos/ICWaves/data/epic/");


% Load channel positions and filter for specified channels
load(fullfile(epic_path, 'scripts/eeg_channel_positions.mat'));
eeg_channels_names = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","FZ","CZ","PZ"];
channel_indices = ismember({eeg_channel_positions.labels}, eeg_channels_names);
eeg_channel_positions = eeg_channel_positions(channel_indices);

% Get data files
brain_files = dir(fullfile(epic_path, 'test/brain', '*timeseries.txt'));
non_brain_files = dir(fullfile(epic_path, 'test/non_brain', '*timeseries.txt'));
weights_dir = fullfile(epic_path, '/test/ica_weights');

% Extract segment info
brain_parsed = cellfun(@(n) sscanf(n, 'segment_%d_component_%d_timeseries.txt')', {brain_files.name}, 'UniformOutput', false);
brain_info = cell2mat(brain_parsed');
non_brain_parsed = cellfun(@(n) sscanf(n, 'segment_%d_component_%d_timeseries.txt')', {non_brain_files.name}, 'UniformOutput', false);
non_brain_info = cell2mat(non_brain_parsed');

% Create output directory
out_dir = fullfile(epic_path, 'raw_data_and_IC_labels');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Process each segment
srate = 256;
segments = unique([brain_info(:,1); non_brain_info(:,1)]);
num_digits = length(num2str(max(segments)));
for i = 1:length(segments)
    fprintf("Processing segment %d out of %d\n", i, length(segments));
    seg_id = segments(i);

    % Load mixing matrix
    icaweights = load(fullfile(weights_dir, sprintf('segment_%d_icamatrix.txt', seg_id)));

    % Get components for this segment
    brain_comps = brain_info(brain_info(:,1) == seg_id, 2);
    non_brain_comps = non_brain_info(non_brain_info(:,1) == seg_id, 2);
    n_comps = size(icaweights, 1);
    n_channels = size(icaweights, 2);

    % Load component data and labels
    comp_data = zeros(n_comps, 10*60*srate); % 10 minutes at srate Hz
    labels = zeros(n_comps, 1);

    for j = 1:length(brain_comps)
        fname = fullfile(epic_path, 'test/brain', sprintf('segment_%d_component_%d_timeseries.txt', seg_id, brain_comps(j)));
        comp_data(brain_comps(j)+1, :) = load(fname);
        labels(brain_comps(j)+1) = 1;
    end

    for j = 1:length(non_brain_comps)
        fname = fullfile(epic_path, 'test/non_brain', sprintf('segment_%d_component_%d_timeseries.txt', seg_id, non_brain_comps(j)));
        comp_data(non_brain_comps(j)+1, :) = load(fname);
        labels(non_brain_comps(j)+1) = 7;
    end

    % Compute raw EEG and create EEG structure
    icasphere = eye(n_channels);
    icawinv = pinv(icaweights * icasphere);
    data = single(icawinv * comp_data);


    EEG = create_eeg_structure(data, eeg_channel_positions, srate, icaweights, icasphere, icawinv);
    EEG = iclabel(EEG);
    noisy_labels = EEG.etc.ic_classification.ICLabel.classifications;
    expert_label_mask = ones(n_comps, 1, 'logical');

    % Save subject file
    out_file = fullfile(out_dir, sprintf(['subj-%0' num2str(num_digits) 'd.mat'], seg_id));
    save(out_file, 'data', 'srate', 'icaweights', 'icasphere', 'noisy_labels', 'labels', 'expert_label_mask', '-v7');
    fprintf('Created %s\n', out_file);
end
