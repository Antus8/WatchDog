function [ADE, FDE] = compute_ADE_FDE(predictions, ground_truths)
    num_samples = length(predictions);
    
    assert(length(ground_truths) == num_samples, 'predictions and gt dimension mismatch');
    cont = 1;

    ADE = cell(6,1);
    FDE = cell(6,1);

    for n_fut = 5:5:30 %Kpos = 30
    
        total_ADE = 0;
        total_FDE = 0;
        total_points = 0;
    
        for i = 1:num_samples
            pred_i = predictions{i};      % 15 x Kpos x 3
            gt_i = ground_truths{i};      % Kpos x 3
    
            % Assumiamo che vogliamo calcolare l'errore rispetto alla prima predizione
            % (es. pred_i(1,:,:) è la previsione fatta al timestep corrente)
            pred = squeeze(pred_i(1,:,:));  % Kpos x 3
    
            if size(pred,1) ~= size(gt_i,1)
                error('Mismatch in numero di timestep tra predizione e ground truth per il sample %d', i);
            end
    
               
            diffs = pred(1:n_fut,:) - gt_i(1:n_fut,:);
            dists = sqrt(sum(diffs.^2, 2));  % distanza euclidea per ogni timestep
    
            total_ADE = total_ADE + sum(dists);
            total_FDE = total_FDE + dists(end);
            total_points = total_points + n_fut;
        end
    
        ADE{cont} = total_ADE / total_points;
        FDE{cont} = total_FDE / num_samples;
        cont=cont+1;
    
    end
end
