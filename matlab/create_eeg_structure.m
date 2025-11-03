function EEG = create_eeg_structure(data, chanlocs, srate, icaweights, icasphere, icawinv)
    % Create minimal EEG structure for ICLabel
    
    EEG = struct();
    EEG.setname = '';
    EEG.data = data;
    EEG.srate = srate;
    EEG.nbchan = size(data, 1);
    EEG.icachansind = 1:size(data, 1);
    EEG.pnts = size(data, 2);
    EEG.trials = 1;
    EEG.chanlocs = chanlocs;
    EEG.ref = 'common';
    EEG.times = 0:1:(size(EEG.data,2)-1);
    EEG.xmin = 0;
    EEG.xmax = EEG.times(end)/srate;
    
    % ICA fields
    EEG.icaweights = icaweights;
    EEG.icasphere = icasphere;
    EEG.icawinv = icawinv;
    EEG.icaact = [];
    
    % Required for ICLabel
    EEG.etc = struct();
end
