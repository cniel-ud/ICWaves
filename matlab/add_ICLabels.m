% Run ICLabel on all subjects and save back the labels in the datafile
eeglab; close;
% Move to folder where this file is located
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end

data_dir = '../data/ds003004';
file_list = dir(fullfile(data_dir, 'sub-*/eeg/*.set'));

for i = 1:length(file_list)
    EEG = pop_loadset(file_list(i).name, file_list(i).folder);
    if isfield(EEG.etc, 'ic_classification')
        fprintf('Found %d labeled ICs in %s\n', ...
            size(EEG.etc.ic_classification.ICLabel.classifications, 1),...
            file_list(i).name)
        continue
    end
    EEG = iclabel(EEG);
    EEG = pop_saveset(EEG, 'savemode', 'resave');
end