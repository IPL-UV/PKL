function computedHSIC = computeHSIC(M, Z, compute_kernel, normalize, nocco)

n_samples = length(M);

if compute_kernel
   % Centering matrix
   H = eye(n_samples) - (1/n_samples)*(ones(n_samples)); 

   % Kernels
   K_m = M*M';
   K_z = Z*Z';
   
   % Centering
   HK_mH = H*K_m*H;
   HK_zH = H*K_z*H;
else
   HK_mH = M;
   HK_zH = Z;
end

% Compute HSIC
HSIC_z_m = 1/(n_samples^2)*trace(HK_mH*HK_zH);
if normalize
    HSIC_m_m = 1/(n_samples^2)*trace(HK_mH*HK_mH); 
    HSIC_z_z = 1/(n_samples^2)*trace(HK_zH*HK_zH);
    HSIC_z_m = HSIC_z_m/sqrt(HSIC_m_m*HSIC_z_z);
end

% Output
computedHSIC = HSIC_z_m;

% Compute NOCCO
if nocco
    epsilon = 10^(-3);
    R_z = HK_zH*pinv(HK_zH + n_samples*epsilon*eye(n_samples));
    R_m = HK_mH*pinv(HK_mH + n_samples*epsilon*eye(n_samples));
    NOCCO = trace(R_z*R_m);
    % Output
    computedHSIC = NOCCO;
end
        
end