function data = load_data_from_csv(csv_file_path)
    
    % loading
    data = readcell(csv_file_path, 'Delimiter', ',');
    num_samples = size(data, 1);
    
    if mod(num_samples, 2) ~= 0
        error('ERROR: the number of lines must be even (input + ground truth)');
    end