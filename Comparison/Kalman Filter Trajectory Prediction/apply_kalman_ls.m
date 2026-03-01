function [x_estimated, v_estimated, future_predictions] = apply_kalman_ls(pos, dt, e, l, Kpos)
    N = size(pos, 1);
    x_estimated = zeros(N, 3);
    v_estimated = zeros(N, 3);
    future_predictions = zeros(N, Kpos, 3); % N timestep, Kpos predizioni future, 3 assi
    
    % Inizializzazione
    x_post = pos(1, :);
    sigma_post = ones(1, 3) * 1e-3;
    meas_noise = 1e-3;
    x_estimated(1, :) = x_post;
    
    for k = 1:N
    
        v_ls = zeros(1, 3);
        if k ~= 1
            % Stima velocità con Least Squares su finestra l
            idx = max(1, k - l + 1):k;
            t = (0:length(idx)-1)' * dt;
            v_ls = zeros(1, 3);
            for d = 1:3
                p = pos(idx, d);
                A = [t ones(length(t),1)];
                coeff = (A' * A) \ (A' * p);
                v_ls(d) = coeff(1);
            end
        end
    
        % Prediction fase Kalman
        x_pri = x_post + v_ls * dt;
    
        % Noise dinamico
        if k > e
            de = abs(pos(k,:) - pos(k - e,:));
        else
            de = [0 0 0];
        end
        a = 0.05; b = 1e-4;
        sys_noise = a * de + b;
        sigma_pri = sigma_post + sys_noise;
    
        % Update fase Kalman
        K = sigma_pri ./ (sigma_pri + meas_noise);
        x_post = x_pri + K .* (pos(k,:) - x_pri);
        sigma_post = (1 - K) .* sigma_pri;
    
        % Salvataggio risultati
        x_estimated(k, :) = x_post;
        v_estimated(k, :) = v_ls;
    
        % Previsione delle prossime Kpos posizioni future
        for f = 1:Kpos
            future_predictions(k, f, :) = x_post + v_ls * dt * f;
        end
    end
end
